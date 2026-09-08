"""Extract MRP (Maximum Retail Price) declaration candidates from OCR detections.

Deterministic and rule-based. No LLM, no network, no guessing. Built with the
same discipline as ``extraction.expiry`` but kept independent of it.

What this module DOES:
  - finds detections whose text carries an MRP label (MRP, M.R.P., MRP:,
    "MRP Rs.", "MRP (incl. of all taxes)", ...);
  - extracts a numeric price either from that same detection (e.g.
    ``MRP:260.00``) or from a nearby detection associated by conservative
    bounding-box proximity (e.g. ``MRP`` + ``₹260.00``);
  - reports the currency ONLY when the OCR clearly shows one (₹ / Rs / INR).

What this module DELIBERATELY DOES NOT DO:
  - it does NOT compare the MRP against anything;
  - it does NOT decide legal compliance;
  - it does NOT silently correct OCR. In particular, a corrupted currency read
    such as ``E260.00`` is NEVER rewritten to ``₹260.00``: the numeric amount
    may still be extracted, but the currency is left unknown and the raw text
    is preserved verbatim, with the suspicion explained in ``note``.

The caller (rules engine / inspector UI) is responsible for any evaluation.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# --- MRP label -------------------------------------------------------------
# Matches MRP / M.R.P. / M R P (optionally spaced or dotted), case-insensitive.
_MRP_LABEL = re.compile(r"M\.?\s*R\.?\s*P\.?", re.IGNORECASE)

# --- Currency --------------------------------------------------------------
_RUPEE_SIGN = "₹"  # ₹
# "Rs" / "Rs." / "INR" as standalone tokens (not embedded in a larger word).
_CURRENCY_WORD = re.compile(r"(?<![A-Za-z])(RS\.?|INR)(?![A-Za-z])", re.IGNORECASE)

# --- Numeric price ---------------------------------------------------------
# A run of digits with optional thousands separators and an optional 1-2 digit
# decimal part. Currency symbols are handled separately, not here.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d{1,2})?")


@dataclass
class MrpCandidate:
    """A single MRP declaration candidate extracted from OCR detections."""

    field_name: str                 # always "MRP"
    raw_text: str                   # exact OCR text of the value's detection, uncorrected
    parsed_value: Optional[float]   # numeric amount if confidently parsed, else None
    currency: Optional[str]         # "₹" / "INR" if clearly present, else None (unknown)
    confidence: Optional[float]     # OCR recognition confidence
    box: Any                        # bounding box, passed through unchanged
    source_image: Optional[str]     # image identifier / path
    uncertain: bool                 # True when the numeric value could not be confidently extracted
    note: str = ""                  # human-readable explanation / assumptions
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


def _has_mrp_label(text: str) -> bool:
    return _MRP_LABEL.search(text) is not None


# --------------------------------------------------------------------------
# Currency + price analysis
# --------------------------------------------------------------------------
def _detect_currency(text: str) -> Optional[str]:
    """Return a currency string only if the OCR clearly shows one, else None."""
    if _RUPEE_SIGN in text:
        return _RUPEE_SIGN
    if _CURRENCY_WORD.search(text):
        return "INR"
    return None


def _analyze_prices(text: str) -> List[Dict[str, Any]]:
    """Find clean, price-like numeric values in ``text``.

    A numeric token counts as *price-like* only when it either carries a
    decimal part (e.g. 260.00) or the text contains a recognized currency
    symbol. This deliberately ignores bare integers such as batch numbers.

    A token immediately followed by a letter (e.g. ``26O`` corruption) is not
    considered a clean value and is skipped rather than silently trimmed.

    Each returned entry describes one value and how its currency was read.
    """
    currency = _detect_currency(text)
    results: List[Dict[str, Any]] = []

    for m in _NUMBER.finditer(text):
        tok = m.group(0)
        start, end = m.span()

        # Reject values glued to a trailing letter (likely OCR contamination).
        right_char = text[end:end + 1]
        if right_char.isalpha():
            continue

        has_decimal = "." in tok
        if not (has_decimal or currency is not None):
            continue  # bare integer without any currency context -> not a price

        preceding = text[:start].rstrip()
        prev_char = preceding[-1] if preceding else ""
        # A stray letter directly before the number (e.g. the "E" in "E260.00")
        # is a suspicious, possibly-misread currency symbol. We do NOT treat it
        # as ₹ — currency stays unknown and we flag it.
        suspicious = currency is None and prev_char.isalpha()

        try:
            value = float(tok.replace(",", ""))
        except ValueError:
            continue

        results.append(
            {
                "value": value,
                "currency": currency,
                "suspicious": suspicious,
                "prev_char": prev_char,
                "raw_token": tok,
            }
        )

    return results


def _currency_note(entry: Dict[str, Any]) -> str:
    if entry["currency"] == _RUPEE_SIGN:
        return "currency ₹ detected"
    if entry["currency"] == "INR":
        return "rupee currency word (Rs/INR) detected"
    if entry["suspicious"]:
        return (
            f"suspicious character {entry['prev_char']!r} precedes the value; "
            f"possibly a misread currency symbol - NOT interpreting it as ₹; "
            f"currency left unknown"
        )
    return "no currency symbol detected; currency left unknown"


# --------------------------------------------------------------------------
# Spatial helpers (self-contained; mirror the expiry extractor in principle)
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


# Kept small so association stays conservative (see extraction.expiry).
_NEARBY_HEIGHT_FACTOR = 1.5


def _find_nearby_prices(
    detections: List[Dict[str, Any]],
    label_index: int,
) -> List[Dict[str, Any]]:
    """Find nearby detections that cleanly contain exactly one price value.

    Qualifies only if the detection (a) is not the label itself, (b) does not
    itself carry an MRP label, (c) contains exactly one clean price value, and
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
        if not text or _has_mrp_label(text):
            continue
        prices = _analyze_prices(text)
        if len(prices) != 1:
            continue  # ambiguous or no clean price in this detection
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
                    "price": prices[0],
                    "distance": distance,
                }
            )
    return matches


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_mrp_candidates(
    detections: List[Dict[str, Any]],
    source_image: Optional[str] = None,
) -> List[MrpCandidate]:
    """Extract MRP declaration candidates from a list of OCR detections.

    Handles both the same-detection case (label and value in one box) and the
    separate-detection case (label box + a nearby value box), associating the
    latter by conservative spatial proximity. When the value cannot be
    confidently extracted, the candidate is returned with ``parsed_value=None``
    and ``uncertain=True`` — never a fabricated number.
    """
    candidates: List[MrpCandidate] = []

    for i, detection in enumerate(detections):
        text = _text_of(detection)
        if not text.strip() or not _has_mrp_label(text):
            continue

        confidence = _confidence_of(detection)
        box = _box_of(detection)
        src = _get(detection, "source_image", default=source_image)

        # (1) Same detection: is there a clean price in the label's own text?
        prices = _analyze_prices(text)

        if len(prices) == 1:
            entry = prices[0]
            candidates.append(
                MrpCandidate(
                    field_name="MRP",
                    raw_text=text,
                    parsed_value=entry["value"],
                    currency=entry["currency"],
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=False,
                    note=_currency_note(entry),
                )
            )
            continue

        if len(prices) > 1:
            values = ", ".join(str(p["value"]) for p in prices)
            candidates.append(
                MrpCandidate(
                    field_name="MRP",
                    raw_text=text,
                    parsed_value=None,
                    currency=_detect_currency(text),
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note=f"multiple price-like values in the same text ({values}); not guessing",
                )
            )
            continue

        # (2) Separate detections: look for a price in a nearby box.
        nearby = _find_nearby_prices(detections, i)

        if len(nearby) == 1:
            match = nearby[0]
            entry = match["price"]
            # Fall back to a currency shown on the label itself (e.g. "MRP Rs.").
            currency = entry["currency"] or _detect_currency(text)
            note = (
                f"price taken from a separate nearby detection "
                f"(gap~{match['distance']:.0f}px); label={text!r}; "
                f"{_currency_note(entry)}"
            )
            candidates.append(
                MrpCandidate(
                    field_name="MRP",
                    raw_text=match["text"],
                    parsed_value=entry["value"],
                    currency=currency,
                    confidence=match["confidence"],
                    box=match["box"],
                    source_image=src,
                    uncertain=False,
                    note=note,
                    label_text=text,
                    label_box=box,
                )
            )
        elif len(nearby) == 0:
            candidates.append(
                MrpCandidate(
                    field_name="MRP",
                    raw_text=text,
                    parsed_value=None,
                    currency=_detect_currency(text),
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note="MRP label found but no usable price in this or a nearby detection",
                    label_text=text,
                    label_box=box,
                )
            )
        else:
            nearby_values = ", ".join(str(m["price"]["value"]) for m in nearby)
            candidates.append(
                MrpCandidate(
                    field_name="MRP",
                    raw_text=text,
                    parsed_value=None,
                    currency=_detect_currency(text),
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note=(
                        f"multiple plausible nearby price values ({nearby_values}); "
                        f"cannot confidently associate one - not guessing"
                    ),
                    label_text=text,
                    label_box=box,
                )
            )

    return candidates
