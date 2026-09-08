"""Extract Manufacturer / Packer / Importer declaration candidates from OCR.

Deterministic and rule-based. No LLM, no network, no guessing. Built with the
same discipline as ``extraction.expiry`` / ``mrp`` / ``quantity`` but kept
independent of them.

What this module DOES:
  - finds detections whose text carries an entity-role label (Manufactured by,
    Manufacturer, Mfd./Mfg. by, Packed by, Packer, Pkd. by, Imported by,
    Importer, Marketed by, ...);
  - preserves the OCR evidence and, when it can be done deterministically,
    separates the label prefix from the entity value;
  - associates a label with a value in a *separate* nearby detection using
    conservative bounding-box proximity.

What this module DELIBERATELY DOES NOT DO:
  - it does NOT decide whether an entity is legally required to be the
    manufacturer / packer / importer;
  - it does NOT decide compliance;
  - it does NOT infer missing information or correct garbled OCR;
  - it does NOT assert a legal role for "Marketed by" (recorded as UNKNOWN).

The caller (rules engine / inspector UI) is responsible for any evaluation.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# Canonical roles.
MANUFACTURER = "MANUFACTURER"
PACKER = "PACKER"
IMPORTER = "IMPORTER"
UNKNOWN = "UNKNOWN"

# --- Entity-role labels ----------------------------------------------------
# Ordered by specificity. The combined "Manufactured & Marketed by" MUST be
# checked before the bare "Marketed by" so the manufacturer role is not lost.
# "Marketed by" alone maps to UNKNOWN: we preserve the declaration but do not
# claim it is the legal manufacturer / packer / importer.
_LABELS: List[Tuple[str, "re.Pattern[str]"]] = [
    (MANUFACTURER, re.compile(r"Manufactured\s*(?:&|and)\s*Marketed\s+by", re.IGNORECASE)),
    (MANUFACTURER, re.compile(r"Manufactured\s+by|Manufacturer", re.IGNORECASE)),
    (MANUFACTURER, re.compile(r"Mf[dg]\.?\s+by", re.IGNORECASE)),
    (PACKER, re.compile(r"Packed\s+by|Pkd\.?\s+by|Packer", re.IGNORECASE)),
    (IMPORTER, re.compile(r"Imported\s+by|Importer", re.IGNORECASE)),
    (UNKNOWN, re.compile(r"Marketed\s+by|Marketer", re.IGNORECASE)),
]

# Leading punctuation/whitespace to peel off the value after the label prefix.
_LEADING_JUNK = re.compile(r"^[\s:.\-–,)]+")


@dataclass
class EntityCandidate:
    """A manufacturer / packer / importer declaration candidate."""

    field_name: str                 # always "ENTITY"
    role: str                       # MANUFACTURER / PACKER / IMPORTER / UNKNOWN
    raw_text: str                   # exact OCR text of the value's detection, uncorrected
    entity_text: Optional[str]      # entity value with the known label prefix removed (raw preserved separately)
    confidence: Optional[float]     # OCR recognition confidence
    box: Any                        # bounding box of the value's detection
    source_image: Optional[str]     # image identifier / path
    uncertain: bool                 # True when a value could not be confidently identified
    note: str = ""                  # human-readable explanation / assumptions
    label_text: Optional[str] = None  # the recognized label token
    label_box: Any = None             # bounding box of the label's detection

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Tolerant detection access
# --------------------------------------------------------------------------
def _get(detection: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in detection and detection[key] is not None:
            return detection[key]
    return default


def _text_of(detection: Dict[str, Any]) -> str:
    text = _get(detection, "text", "rec_text", default="")
    return text if isinstance(text, str) else ""


def _box_of(detection: Dict[str, Any]) -> Any:
    return _get(detection, "box", "bbox", "bounding_box", "poly")


def _confidence_of(detection: Dict[str, Any]) -> Any:
    return _get(detection, "confidence", "score", "rec_score")


# --------------------------------------------------------------------------
# Label + entity-text helpers
# --------------------------------------------------------------------------
def _match_label(text: str) -> Optional[Tuple[str, str, int]]:
    """Return (role, matched_label_text, match_end) for the first label found."""
    for role, pattern in _LABELS:
        m = pattern.search(text)
        if m:
            return role, m.group(0), m.end()
    return None


def _strip_label_prefix(text: str, label_end: int) -> str:
    """Return the value portion after the label, minus leading punctuation.

    Only leading label punctuation/whitespace is removed. Internal wording and
    trailing content are preserved verbatim (no OCR cleanup / correction).
    """
    tail = text[label_end:]
    tail = _LEADING_JUNK.sub("", tail)
    return tail.strip()


def _is_plausible_entity(text: str) -> bool:
    """Conservative check that ``text`` looks like an entity value at all.

    Requires a little alphabetic content. This does NOT judge correctness — a
    garbled-but-present name still counts (and is preserved verbatim).
    """
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    letters = sum(1 for ch in stripped if ch.isalpha())
    return letters >= 2


# --------------------------------------------------------------------------
# Spatial helpers (self-contained; mirror the other extractors in principle)
# --------------------------------------------------------------------------
def _aabb(box: Any) -> Optional[tuple[float, float, float, float]]:
    if not box:
        return None
    try:
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x0, y0, x1, y1 = box
            return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError, IndexError):
        return None


def _box_gap_distance(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


# Kept small so association stays conservative (see the sibling extractors).
_NEARBY_HEIGHT_FACTOR = 1.5


def _find_nearby_entities(
    detections: List[Dict[str, Any]],
    label_index: int,
) -> List[Dict[str, Any]]:
    """Find nearby detections that plausibly hold an entity value.

    Qualifies only if the detection (a) is not the label itself, (b) does not
    itself carry an entity-role label, (c) contains plausible entity text, and
    (d) lies within a small distance of the label box.
    """
    label_aabb = _aabb(_box_of(detections[label_index]))
    if label_aabb is None:
        return []
    label_height = max(1.0, label_aabb[3] - label_aabb[1])
    threshold = _NEARBY_HEIGHT_FACTOR * label_height

    matches: List[Dict[str, Any]] = []
    for j, det in enumerate(detections):
        if j == label_index:
            continue
        text = _text_of(det).strip()
        if not text or _match_label(text) is not None:
            continue
        if not _is_plausible_entity(text):
            continue
        aabb = _aabb(_box_of(det))
        if aabb is None:
            continue
        distance = _box_gap_distance(label_aabb, aabb)
        if distance <= threshold:
            matches.append(
                {
                    "text": _text_of(det),
                    "confidence": _confidence_of(det),
                    "box": _box_of(det),
                    "distance": distance,
                }
            )
    return matches


def _role_note(role: str) -> str:
    if role == UNKNOWN:
        return (
            "label recognized but role left UNKNOWN "
            "(not asserting manufacturer/packer/importer, e.g. 'Marketed by')"
        )
    return ""


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_entity_candidates(
    detections: List[Dict[str, Any]],
    source_image: Optional[str] = None,
) -> List[EntityCandidate]:
    """Extract manufacturer / packer / importer candidates from OCR detections.

    Handles the same-detection case (label and value together) and the
    separate-detection case (label box + a nearby value box), associating the
    latter by conservative spatial proximity. When a value cannot be
    confidently identified, an uncertain candidate is returned — never a guess.
    """
    candidates: List[EntityCandidate] = []

    for i, detection in enumerate(detections):
        text = _text_of(detection)
        if not text.strip():
            continue

        matched = _match_label(text)
        if matched is None:
            continue
        role, label_text, label_end = matched

        confidence = _confidence_of(detection)
        box = _box_of(detection)
        src = _get(detection, "source_image", default=source_image)

        # (1) Same detection: is there a plausible value after the label?
        entity_text = _strip_label_prefix(text, label_end)
        if _is_plausible_entity(entity_text):
            note = "entity parsed from same detection (label prefix removed for entity_text; raw_text preserved)"
            role_note = _role_note(role)
            if role_note:
                note += f"; {role_note}"
            candidates.append(
                EntityCandidate(
                    field_name="ENTITY",
                    role=role,
                    raw_text=text,
                    entity_text=entity_text,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=False,
                    note=note,
                    label_text=label_text,
                    label_box=box,
                )
            )
            continue

        # (2) Separate detections: look for a value in a nearby box.
        nearby = _find_nearby_entities(detections, i)

        if len(nearby) == 1:
            match = nearby[0]
            note = (
                f"entity taken from a separate nearby detection "
                f"(gap~{match['distance']:.0f}px); label={label_text!r}"
            )
            role_note = _role_note(role)
            if role_note:
                note += f"; {role_note}"
            candidates.append(
                EntityCandidate(
                    field_name="ENTITY",
                    role=role,
                    raw_text=match["text"],
                    entity_text=match["text"].strip(),
                    confidence=match["confidence"],
                    box=match["box"],
                    source_image=src,
                    uncertain=False,
                    note=note,
                    label_text=label_text,
                    label_box=box,
                )
            )
        elif len(nearby) == 0:
            candidates.append(
                EntityCandidate(
                    field_name="ENTITY",
                    role=role,
                    raw_text=text,
                    entity_text=None,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note="entity label found but no plausible entity value in this or a nearby detection",
                    label_text=label_text,
                    label_box=box,
                )
            )
        else:
            nearby_texts = ", ".join(repr(m["text"]) for m in nearby)
            candidates.append(
                EntityCandidate(
                    field_name="ENTITY",
                    role=role,
                    raw_text=text,
                    entity_text=None,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note=(
                        f"multiple plausible nearby entity values ({nearby_texts}); "
                        f"cannot confidently associate one - not guessing"
                    ),
                    label_text=label_text,
                    label_box=box,
                )
            )

    return candidates
