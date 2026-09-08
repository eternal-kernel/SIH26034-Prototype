"""Extract NET QUANTITY / NET CONTENT declaration candidates from OCR detections.

Deterministic and rule-based. No LLM, no network, no guessing. Built with the
same discipline as ``extraction.expiry`` and ``extraction.mrp`` but kept
independent of them.

What this module DOES:
  - finds detections whose text carries a net-quantity label (NET CONTENT,
    NET QUANTITY, NET WT / NET WT. / NET WEIGHT, ...);
  - extracts every clearly parseable quantity representation, either from the
    same detection (e.g. ``Net Wt. 500 g``) or from a nearby detection
    associated by conservative bounding-box proximity (e.g. ``Net Content:`` +
    ``1L/553g``);
  - preserves ALL representations found in a declaration.

What this module DELIBERATELY DOES NOT DO:
  - it does NOT decide which representation is the legally applicable one;
  - it does NOT perform minimum-quantity / MPE / unit-price calculations;
  - it does NOT compare against physical measurements;
  - it does NOT decide legal compliance;
  - it does NOT silently correct OCR or invent a missing digit / unit. If a
    number appears without a recognizable unit, no unit is fabricated and the
    candidate is flagged uncertain.

The caller (rules engine / inspector UI) is responsible for any evaluation.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# --- Net-quantity label ----------------------------------------------------
# NET CONTENT / NET QUANTITY / NET QTY / NET WEIGHT / NET WT / NET WT.
_NET_LABEL = re.compile(
    r"NET\s*(?:CONTENT|QUANTITY|QTY|WEIGHT|WT)\.?", re.IGNORECASE
)

# --- Quantity representations ----------------------------------------------
# A number followed by a supported unit. Multi-char units are listed before
# single-char ones so, e.g., "ml"/"kg"/"mg" win over "l"/"g". The trailing
# (?![A-Za-z]) guard rejects units glued to more letters (e.g. "gm", "Litre"),
# which we treat as suspicious rather than silently reinterpreting.
_QUANTITY = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|mg|ml|g|l)(?![A-Za-z])", re.IGNORECASE
)

# Normalize the matched unit while keeping the raw OCR text intact elsewhere.
_UNIT_NORMALIZE = {"g": "g", "kg": "kg", "mg": "mg", "ml": "ml", "l": "L"}


@dataclass
class QuantityCandidate:
    """A net-quantity declaration candidate extracted from OCR detections."""

    field_name: str                     # always "NET_QUANTITY"
    raw_text: str                       # exact OCR text of the value's detection
    quantities: List[Dict[str, Any]] = field(default_factory=list)  # every parsed representation
    confidence: Optional[float] = None  # OCR recognition confidence
    box: Any = None                     # bounding box, passed through unchanged
    source_image: Optional[str] = None  # image identifier / path
    uncertain: bool = False             # True when no representation could be confidently parsed
    note: str = ""                      # human-readable explanation / assumptions
    # Preserve the label's own evidence when the value came from a separate box.
    label_text: Optional[str] = None
    label_box: Any = None

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


def _has_net_label(text: str) -> bool:
    return _NET_LABEL.search(text) is not None


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


# --------------------------------------------------------------------------
# Quantity parsing
# --------------------------------------------------------------------------
def _parse_quantities(text: str) -> List[Dict[str, Any]]:
    """Return every clearly parseable (value, unit) representation in ``text``.

    Each entry keeps the raw matched substring so the original OCR is preserved
    even though the unit is normalized. Multiple representations (e.g. the two
    in ``1L/553g``) are ALL returned, in order — none is dropped or chosen.
    """
    out: List[Dict[str, Any]] = []
    for m in _QUANTITY.finditer(text):
        number, unit_raw = m.group(1), m.group(2)
        out.append(
            {
                "value": float(number),
                "unit": _UNIT_NORMALIZE[unit_raw.lower()],
                "raw": m.group(0),
            }
        )
    return out


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


# Kept small so association stays conservative (see extraction.expiry / mrp).
_NEARBY_HEIGHT_FACTOR = 1.5


def _find_nearby_quantities(
    detections: List[Dict[str, Any]],
    label_index: int,
) -> List[Dict[str, Any]]:
    """Find nearby detections that contain at least one parseable quantity.

    Qualifies only if the detection (a) is not the label itself, (b) does not
    itself carry a net-quantity label, (c) contains >= 1 parseable quantity,
    and (d) lies within a small distance of the label box. A single detection
    may hold several representations (e.g. ``1L/553g``) — that still counts as
    ONE nearby detection, so it is not treated as ambiguous.
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
        if not text or _has_net_label(text):
            continue
        quantities = _parse_quantities(text)
        if not quantities:
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
                    "quantities": quantities,
                    "distance": distance,
                }
            )
    return matches


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_quantity_candidates(
    detections: List[Dict[str, Any]],
    source_image: Optional[str] = None,
) -> List[QuantityCandidate]:
    """Extract net-quantity candidates from a list of OCR detections.

    Handles the same-detection case (label and value together) and the
    separate-detection case (label box + a nearby value box), associating the
    latter by conservative spatial proximity. All clearly parsed
    representations are preserved; none is selected as "the" quantity. When no
    quantity can be confidently parsed, the candidate is returned with
    ``quantities=[]`` and ``uncertain=True`` — never fabricated.
    """
    candidates: List[QuantityCandidate] = []

    for i, detection in enumerate(detections):
        text = _text_of(detection)
        if not text.strip() or not _has_net_label(text):
            continue

        confidence = _confidence_of(detection)
        box = _box_of(detection)
        src = _get(detection, "source_image", default=source_image)

        # (1) Same detection: are there quantities in the label's own text?
        own_quantities = _parse_quantities(text)
        if own_quantities:
            candidates.append(
                QuantityCandidate(
                    field_name="NET_QUANTITY",
                    raw_text=text,
                    quantities=own_quantities,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=False,
                    note=(
                        f"{len(own_quantities)} representation(s) parsed from the "
                        f"same detection; all preserved"
                    ),
                )
            )
            continue

        # A bare number with no recognizable unit lives right here in the label
        # detection -> do not go hunting elsewhere and do not invent a unit.
        if _has_digit(text):
            candidates.append(
                QuantityCandidate(
                    field_name="NET_QUANTITY",
                    raw_text=text,
                    quantities=[],
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note="number present but no recognizable unit; not inventing a unit",
                    label_text=text,
                    label_box=box,
                )
            )
            continue

        # (2) Separate detections: look for quantities in a nearby box.
        nearby = _find_nearby_quantities(detections, i)

        if len(nearby) == 1:
            match = nearby[0]
            candidates.append(
                QuantityCandidate(
                    field_name="NET_QUANTITY",
                    raw_text=match["text"],
                    quantities=match["quantities"],
                    confidence=match["confidence"],
                    box=match["box"],
                    source_image=src,
                    uncertain=False,
                    note=(
                        f"quantity taken from a separate nearby detection "
                        f"(gap~{match['distance']:.0f}px); label={text!r}; "
                        f"{len(match['quantities'])} representation(s) preserved"
                    ),
                    label_text=text,
                    label_box=box,
                )
            )
        elif len(nearby) == 0:
            candidates.append(
                QuantityCandidate(
                    field_name="NET_QUANTITY",
                    raw_text=text,
                    quantities=[],
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note="net-quantity label found but no usable quantity in this or a nearby detection",
                    label_text=text,
                    label_box=box,
                )
            )
        else:
            summaries = "; ".join(
                "+".join(f"{q['value']}{q['unit']}" for q in m["quantities"])
                for m in nearby
            )
            candidates.append(
                QuantityCandidate(
                    field_name="NET_QUANTITY",
                    raw_text=text,
                    quantities=[],
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note=(
                        f"multiple plausible nearby quantity detections ({summaries}); "
                        f"cannot confidently associate one - not guessing"
                    ),
                    label_text=text,
                    label_box=box,
                )
            )

    return candidates
