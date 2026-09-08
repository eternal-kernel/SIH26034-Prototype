"""Extract Generic / Common / Product Name declaration candidates from OCR.

Deterministic and rule-based. No LLM, no network, no guessing. Built with the
same discipline as the sibling extractors (expiry / mrp / quantity / entity)
but kept independent of them.

Philosophy: be CONSERVATIVE. A product-name candidate is produced only when an
EXPLICIT name label is present (Common Name, Generic Name, Product Name, Name
of Product, Generic Description, Description of Product). Brand / marketing
wording on its own is NOT treated as the generic name, and values that look
like other declarations (MRP, quantity, dates, manufacturer, address, phone,
URL, email, batch/barcode) are rejected.

What this module DELIBERATELY DOES NOT DO:
  - it does NOT decide legal compliance or whether the name satisfies a
    declaration requirement;
  - it does NOT infer a product category from unrelated text;
  - it does NOT silently correct OCR or invent missing words;
  - it does NOT pick a "better" name out of arbitrary marketing text.

The caller (rules engine / inspector UI) is responsible for any evaluation.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Explicit product-name labels ------------------------------------------
_NAME_LABEL = re.compile(
    r"(Common\s+Name|Generic\s+Name|Product\s+Name|Name\s+of\s+Product|"
    r"Generic\s+Description|Description\s+of\s+Product)",
    re.IGNORECASE,
)

# Leading punctuation/whitespace to peel off the value after the label prefix.
_LEADING_JUNK = re.compile(r"^[\s:.\-–,)]+")

# --- "Looks like some OTHER declaration" rejectors -------------------------
# If any of these match, the text is not accepted as a product name.
_OTHER_DECLARATION = [
    ("mrp/price", re.compile(r"M\.?\s*R\.?\s*P\.?|₹|(?<![A-Za-z])RS\.?(?![A-Za-z])|\bINR\b", re.IGNORECASE)),
    ("quantity", re.compile(r"NET\s*(?:CONTENT|QUANTITY|QTY|WEIGHT|WT)|(?<!\w)\d+(?:\.\d+)?\s*(?:kg|mg|ml|g|l)(?![A-Za-z])", re.IGNORECASE)),
    ("date/expiry/batch", re.compile(r"\bEXP\b|EXPIRY|USE\s*-?\s*BY|BEST\s*-?\s*BEFORE|MF[DG]\b|\bPKD\b|BATCH|\bBN\b|\d{1,2}[/\-.][A-Za-z0-9]{2,}[/\-.]\d{2,4}", re.IGNORECASE)),
    ("entity", re.compile(r"Manufactured|Manufacturer|Packed\b|Packer|Imported|Importer|Marketed|Marketer|Mf[dg]\.?\s+by|Pkd\.?\s+by", re.IGNORECASE)),
    ("address", re.compile(r"\b\d{6}\b|\bRoad\b|\bStreet\b|\bNagar\b|\bIndia\b|\bDist\b|\bP\.?O\.?\b", re.IGNORECASE)),
    ("phone", re.compile(r"Toll\s*free|\b1800\b|\b\d{10}\b|\bLic\.?\s*No\b", re.IGNORECASE)),
    ("url", re.compile(r"https?://|www\.|\.com\b|\.coop\b|\.in\b|\.org\b", re.IGNORECASE)),
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+", re.IGNORECASE)),
]


@dataclass
class ProductNameCandidate:
    """A generic/common/product-name declaration candidate."""

    field_name: str                  # always "PRODUCT_NAME"
    raw_text: str                    # exact OCR text of the value's detection, uncorrected
    product_name: Optional[str]      # value after an explicit label (raw preserved separately)
    confidence: Optional[float]      # OCR recognition confidence
    box: Any                         # bounding box of the value's detection
    source_image: Optional[str]      # image identifier / path
    uncertain: bool                  # True when a value could not be confidently identified
    note: str = ""                   # human-readable explanation / assumptions
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
# Label + value helpers
# --------------------------------------------------------------------------
def _match_label(text: str) -> Optional[Tuple[str, int]]:
    """Return (matched_label_text, match_end) for the first name label found."""
    m = _NAME_LABEL.search(text)
    if m:
        return m.group(0), m.end()
    return None


def _strip_label_prefix(text: str, label_end: int) -> str:
    tail = _LEADING_JUNK.sub("", text[label_end:])
    return tail.strip()


def _looks_like_other_declaration(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in _OTHER_DECLARATION)


def _is_plausible_product_name(text: str) -> bool:
    """Conservative check that ``text`` reads like a product/common name.

    Requires word-like content and rejects anything that matches another
    declaration type. Does NOT judge correctness — a garbled-but-word-like
    name still counts (and is preserved verbatim).
    """
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    if _looks_like_other_declaration(stripped):
        return False
    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters < 3:
        return False
    # Product names are predominantly letters/spaces, not codes or numbers.
    alpha_space = sum(1 for ch in stripped if ch.isalpha() or ch.isspace())
    return (alpha_space / len(stripped)) >= 0.6


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


def _find_nearby_names(
    detections: List[Dict[str, Any]],
    label_index: int,
) -> List[Dict[str, Any]]:
    """Find nearby detections that plausibly hold a product-name value.

    Qualifies only if the detection (a) is not the label itself, (b) does not
    itself carry a name label, (c) reads like a plausible product name, and
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
        if not _is_plausible_product_name(text):
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


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_product_name_candidates(
    detections: List[Dict[str, Any]],
    source_image: Optional[str] = None,
) -> List[ProductNameCandidate]:
    """Extract product-name candidates from OCR detections.

    A candidate is produced only where an EXPLICIT name label is present.
    Handles the same-detection case (label + value together) and the
    separate-detection case (label box + a nearby value box), associating the
    latter by conservative spatial proximity. When a value cannot be
    confidently identified, an uncertain candidate is returned — never a guess.
    """
    candidates: List[ProductNameCandidate] = []

    for i, detection in enumerate(detections):
        text = _text_of(detection)
        if not text.strip():
            continue

        matched = _match_label(text)
        if matched is None:
            continue  # no explicit product-name label -> not extracted
        label_text, label_end = matched

        confidence = _confidence_of(detection)
        box = _box_of(detection)
        src = _get(detection, "source_image", default=source_image)

        # (1) Same detection: is there a plausible value after the label?
        value = _strip_label_prefix(text, label_end)
        if _is_plausible_product_name(value):
            candidates.append(
                ProductNameCandidate(
                    field_name="PRODUCT_NAME",
                    raw_text=text,
                    product_name=value,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=False,
                    note="product name parsed from same detection (label prefix removed; raw_text preserved)",
                    label_text=label_text,
                    label_box=box,
                )
            )
            continue

        # If a value is present but does not read like a product name, do not
        # go hunting elsewhere and do not accept it.
        if value:
            candidates.append(
                ProductNameCandidate(
                    field_name="PRODUCT_NAME",
                    raw_text=text,
                    product_name=None,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note="name label found but the adjacent value does not read like a product name",
                    label_text=label_text,
                    label_box=box,
                )
            )
            continue

        # (2) Separate detections: look for a value in a nearby box.
        nearby = _find_nearby_names(detections, i)

        if len(nearby) == 1:
            match = nearby[0]
            candidates.append(
                ProductNameCandidate(
                    field_name="PRODUCT_NAME",
                    raw_text=match["text"],
                    product_name=match["text"].strip(),
                    confidence=match["confidence"],
                    box=match["box"],
                    source_image=src,
                    uncertain=False,
                    note=(
                        f"product name taken from a separate nearby detection "
                        f"(gap~{match['distance']:.0f}px); label={label_text!r}"
                    ),
                    label_text=label_text,
                    label_box=box,
                )
            )
        elif len(nearby) == 0:
            candidates.append(
                ProductNameCandidate(
                    field_name="PRODUCT_NAME",
                    raw_text=text,
                    product_name=None,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note="name label found but no plausible product-name value in this or a nearby detection",
                    label_text=label_text,
                    label_box=box,
                )
            )
        else:
            nearby_texts = ", ".join(repr(m["text"]) for m in nearby)
            candidates.append(
                ProductNameCandidate(
                    field_name="PRODUCT_NAME",
                    raw_text=text,
                    product_name=None,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note=(
                        f"multiple plausible nearby product-name values ({nearby_texts}); "
                        f"cannot confidently associate one - not guessing"
                    ),
                    label_text=label_text,
                    label_box=box,
                )
            )

    return candidates
