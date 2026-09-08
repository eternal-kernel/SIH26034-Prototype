"""Extract EXPIRY / USE-BY / BEST-BEFORE date candidates from OCR detections.

Deterministic and rule-based. No LLM, no network, no guessing.

What this module DOES:
  - finds detections whose text carries an expiry-type label
    (EXP, EXPIRY, USE BY, USE-BY, BEST BEFORE, BEST-BEFORE);
  - parses a date either from that same detection (e.g. the real Amul label
    ``EXP:17/JAN/27``) or, when the label stands alone, from a *nearby*
    detection associated by conservative spatial proximity (e.g. ``EXP`` in
    one box and ``17/JAN/27`` in the box beside it);
  - returns a structured candidate for each match.

What this module DELIBERATELY DOES NOT DO:
  - it does NOT decide whether a product is expired;
  - it does NOT compare the parsed date against today's date;
  - it does NOT silently correct noisy OCR — if a date cannot be parsed with
    confidence, the candidate is returned with ``parsed_date = None`` and
    ``uncertain = True`` plus a human-readable note.

The caller (rules engine / inspector UI) is responsible for any evaluation.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, List, Optional

# --- Labels ----------------------------------------------------------------
# Ordered longest / most-specific first so that, e.g., "EXPIRY" is classified
# as EXPIRY before the short "EXP" alias is considered. Each entry maps a
# regular expression to the canonical field name it represents.
_LABEL_PATTERNS = [
    (re.compile(r"BEST[\s\-]?BEFORE", re.IGNORECASE), "BEST_BEFORE"),
    (re.compile(r"USE[\s\-]?BY", re.IGNORECASE), "USE_BY"),
    (re.compile(r"EXPIRY", re.IGNORECASE), "EXPIRY"),
    (re.compile(r"\bEXP\b", re.IGNORECASE), "EXPIRY"),
]

# --- Date formats ----------------------------------------------------------
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Day / month-name / year  ->  e.g. 17/JAN/27  or  17-JAN-2027
_DATE_ALPHA = re.compile(
    r"(?P<d>\d{1,2})\s*[/\-.]\s*(?P<m>[A-Za-z]{3,9})\s*[/\-.]\s*(?P<y>\d{2,4})"
)
# Day / month-number / year  ->  e.g. 17/01/27  or  17-01-2027
_DATE_NUMERIC = re.compile(
    r"(?P<d>\d{1,2})\s*[/\-.]\s*(?P<m>\d{1,2})\s*[/\-.]\s*(?P<y>\d{2,4})"
)


@dataclass
class DateCandidate:
    """A single expiry-type date candidate extracted from one OCR detection."""

    field_name: str                 # canonical label: EXPIRY / USE_BY / BEST_BEFORE
    raw_text: str                   # exact OCR text, uncorrected
    parsed_date: Optional[str]      # ISO 8601 (YYYY-MM-DD) if confidently parsed, else None
    confidence: Optional[float]     # OCR recognition confidence for the detection
    box: Any                        # bounding box, passed through unchanged
    source_image: Optional[str]     # image identifier / path this came from
    uncertain: bool                 # True when the date could not be parsed with confidence
    note: str = ""                  # human-readable explanation / assumptions
    # When the date came from a *separate* nearby detection, these preserve the
    # label's own evidence (they stay None for the same-detection case).
    label_text: Optional[str] = None
    label_box: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_year(raw_year: str) -> int:
    """Expand a 2-digit year to 20YY; leave 4-digit years as-is.

    This is an explicit, documented assumption — not a silent correction.
    """
    if len(raw_year) <= 2:
        return 2000 + int(raw_year)
    return int(raw_year)


def _parse_date(text: str) -> tuple[Optional[str], str]:
    """Try to parse a supported date out of ``text``.

    Returns ``(iso_date_or_None, note)``. A ``None`` date means the text did
    not match a known format (or matched but was not a valid calendar date);
    the note explains why so the caller can surface it to the inspector.
    """
    notes: List[str] = []

    # 1) Alphabetic month (the style on the real Amul label: 17/JAN/27).
    m = _DATE_ALPHA.search(text)
    if m:
        month_key = m.group("m")[:3].upper()
        if month_key not in _MONTHS:
            return None, f"unrecognized month token '{m.group('m')}'"
        day = int(m.group("d"))
        month = _MONTHS[month_key]
        raw_year = m.group("y")
        year = _normalize_year(raw_year)
        if len(raw_year) <= 2:
            notes.append(f"2-digit year '{raw_year}' assumed 20{raw_year}")
        try:
            iso = date(year, month, day).isoformat()
        except ValueError as exc:
            return None, f"invalid calendar date: {exc}"
        return iso, "; ".join(notes)

    # 2) Fully numeric date (kept conservative: DAY/MONTH/YEAR order).
    m = _DATE_NUMERIC.search(text)
    if m:
        day = int(m.group("d"))
        month = int(m.group("m"))
        raw_year = m.group("y")
        year = _normalize_year(raw_year)
        if len(raw_year) <= 2:
            notes.append(f"2-digit year '{raw_year}' assumed 20{raw_year}")
        notes.append("numeric date assumed DAY/MONTH/YEAR order")
        try:
            iso = date(year, month, day).isoformat()
        except ValueError as exc:
            return None, f"invalid calendar date: {exc}"
        return iso, "; ".join(notes)

    return None, "no recognizable date pattern"


def _get(detection: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Fetch the first present key from a detection dict (tolerant of aliases)."""
    for key in keys:
        if key in detection and detection[key] is not None:
            return detection[key]
    return default


def _text_of(detection: Dict[str, Any]) -> str:
    text = _get(detection, "text", "rec_text", default="")
    return text if isinstance(text, str) else ""


def _box_of(detection: Dict[str, Any]) -> Any:
    return _get(detection, "box", "bbox", "bounding_box", "poly")


def _match_label(text: str) -> Optional[str]:
    """Return the canonical expiry-type field for ``text``, or None."""
    for pattern, canonical in _LABEL_PATTERNS:
        if pattern.search(text):
            return canonical
    return None


def _aabb(box: Any) -> Optional[tuple[float, float, float, float]]:
    """Axis-aligned bounding box (x0, y0, x1, y1) from a polygon or flat box.

    Accepts either a list of [x, y] points (PaddleOCR polygon) or a flat
    [x0, y0, x1, y1]. Returns None if the box is missing/unrecognized.
    """
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
    """Edge-to-edge gap distance between two AABBs (0 if they overlap)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


# How close (relative to the label's height) a separate detection must be for
# it to count as a plausible date for that label. Deliberately small so the
# association stays conservative — a date somewhere else on the pack will not
# be pulled in.
_NEARBY_HEIGHT_FACTOR = 1.5


def _find_nearby_dates(
    detections: List[Dict[str, Any]],
    label_index: int,
) -> List[Dict[str, Any]]:
    """Find date-bearing detections spatially near the label detection.

    A detection qualifies only if it (a) is not the label itself, (b) does not
    itself carry an expiry-type label (so we never steal another field's date),
    (c) parses to a valid date, and (d) lies within a small distance of the
    label box. Returns each match with its parsed distance for reporting.
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
        if not text:
            continue
        if _match_label(text) is not None:
            continue  # belongs to its own labeled field
        iso, _ = _parse_date(text)
        if iso is None:
            continue  # only date-bearing detections can be associated
        aabb = _aabb(_box_of(det))
        if aabb is None:
            continue
        distance = _box_gap_distance(label_aabb, aabb)
        if distance <= threshold:
            matches.append(
                {
                    "text": _text_of(det),
                    "confidence": _get(det, "confidence", "score", "rec_score"),
                    "box": _box_of(det),
                    "parsed_date": iso,
                    "distance": distance,
                }
            )
    return matches


def extract_expiry_candidates(
    detections: List[Dict[str, Any]],
    source_image: Optional[str] = None,
) -> List[DateCandidate]:
    """Extract expiry-type date candidates from a list of OCR detections.

    Each detection is a dict with (tolerant) keys:
      - text: recognized string (``text`` / ``rec_text``)
      - confidence: recognition score (``confidence`` / ``score`` / ``rec_score``)
      - box: bounding box (``box`` / ``bbox`` / ``bounding_box`` / ``poly``)
      - source_image: optional per-detection image id/path

    ``source_image`` (the function argument) is used as a fallback when a
    detection does not carry its own.

    Two shapes are handled:
      1. Same detection — the label and date share one OCR box
         (e.g. ``EXP:17/JAN/27``). Parsed directly from that text.
      2. Separate detections — a label box (e.g. ``EXP``) with the date in a
         nearby box (e.g. ``17/JAN/27``). Associated by conservative spatial
         proximity: exactly one nearby date -> associate; zero or several ->
         an uncertain candidate (never a guess).

    Returns a list of :class:`DateCandidate`, one per detection that carries an
    expiry-type label.
    """
    candidates: List[DateCandidate] = []

    for i, detection in enumerate(detections):
        text = _text_of(detection)
        if not text.strip():
            continue

        field_name = _match_label(text)
        if field_name is None:
            continue  # not an expiry-type detection

        confidence = _get(detection, "confidence", "score", "rec_score")
        box = _box_of(detection)
        src = _get(detection, "source_image", default=source_image)

        # (1) Same-detection: the label's own text already contains the date.
        iso_date, note = _parse_date(text)
        if iso_date is not None:
            candidates.append(
                DateCandidate(
                    field_name=field_name,
                    raw_text=text,
                    parsed_date=iso_date,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=False,
                    note=note,
                )
            )
            continue

        # (2) Separate detections: look for a date in a nearby box.
        nearby = _find_nearby_dates(detections, i)

        if len(nearby) == 1:
            match = nearby[0]
            match_note = _parse_date(match["text"])[1]
            note = (
                f"date taken from a separate nearby detection "
                f"(gap~{match['distance']:.0f}px); label={text!r}"
            )
            if match_note:
                note += f"; {match_note}"
            candidates.append(
                DateCandidate(
                    field_name=field_name,
                    raw_text=match["text"],
                    parsed_date=match["parsed_date"],
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
                DateCandidate(
                    field_name=field_name,
                    raw_text=text,
                    parsed_date=None,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note="label found but no parseable date in this or a nearby detection",
                    label_text=text,
                    label_box=box,
                )
            )
        else:
            nearby_texts = ", ".join(repr(m["text"]) for m in nearby)
            candidates.append(
                DateCandidate(
                    field_name=field_name,
                    raw_text=text,
                    parsed_date=None,
                    confidence=confidence,
                    box=box,
                    source_image=src,
                    uncertain=True,
                    note=(
                        f"multiple plausible nearby dates ({nearby_texts}); "
                        f"cannot confidently associate one - not guessing"
                    ),
                    label_text=text,
                    label_box=box,
                )
            )

    return candidates
