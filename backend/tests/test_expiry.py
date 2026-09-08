"""Unit tests for extraction.expiry.

Runnable two ways:
  - directly:   python tests/test_expiry.py     (from the backend/ directory)
  - via pytest: pytest tests/test_expiry.py      (pytest is optional)

The primary test uses the real Amul OCR detection ``EXP:17/JAN/27`` with the
confidence value observed in the diagnostic run (0.9631).
"""

import os
import sys

# Make the backend/ directory importable regardless of how this file is run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.expiry import extract_expiry_candidates  # noqa: E402


def test_amul_exp_date_parsed():
    """EXP:17/JAN/27 (conf 0.9631) parses to 2027-01-17 and is not uncertain."""
    detections = [
        {
            "text": "EXP:17/JAN/27",
            "confidence": 0.9631,
            "box": [[250, 978], [542, 971], [543, 1002], [251, 1009]],
            "source_image": "amul_icecream.jpeg",
        }
    ]

    candidates = extract_expiry_candidates(detections)

    assert len(candidates) == 1, f"expected exactly 1 candidate, got {len(candidates)}"
    c = candidates[0]
    assert c.field_name == "EXPIRY", c.field_name
    assert c.raw_text == "EXP:17/JAN/27", c.raw_text
    assert c.parsed_date == "2027-01-17", c.parsed_date
    assert c.uncertain is False, c.note
    assert c.confidence == 0.9631
    assert c.box == [[250, 978], [542, 971], [543, 1002], [251, 1009]]
    assert c.source_image == "amul_icecream.jpeg"
    return c


def test_unparseable_label_is_uncertain_not_invented():
    """A label with no parseable date must be flagged uncertain, not fabricated."""
    detections = [
        {
            "text": "Expiry (Exp). Batch No (BN), see body",
            "confidence": 0.9042,
            "box": [[51, 891], [617, 913]],
        }
    ]

    candidates = extract_expiry_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.field_name == "EXPIRY"
    assert c.parsed_date is None, "must not invent a date"
    assert c.uncertain is True
    return c


def test_separate_nearby_detections_associated():
    """B. 'EXP' + a nearby '17/JAN/27' associate into one confident candidate."""
    detections = [
        {
            "text": "EXP",
            "confidence": 0.96,
            "box": [[250, 978], [300, 975], [301, 1002], [251, 1005]],
            "source_image": "synthetic.png",
        },
        {
            "text": "17/JAN/27",
            "confidence": 0.96,
            "box": [[305, 978], [390, 975], [391, 1002], [306, 1005]],
            "source_image": "synthetic.png",
        },
    ]

    candidates = extract_expiry_candidates(detections)

    assert len(candidates) == 1, f"expected 1 candidate, got {len(candidates)}"
    c = candidates[0]
    assert c.field_name == "EXPIRY", c.field_name
    assert c.raw_text == "17/JAN/27", c.raw_text          # value comes from the date box
    assert c.parsed_date == "2027-01-17", c.parsed_date
    assert c.uncertain is False, c.note
    assert c.confidence == 0.96
    assert c.box == [[305, 978], [390, 975], [391, 1002], [306, 1005]]  # date box preserved
    assert c.label_text == "EXP", c.label_text            # label evidence preserved
    assert c.label_box == [[250, 978], [300, 975], [301, 1002], [251, 1005]]
    assert c.source_image == "synthetic.png"
    return c


def test_label_with_only_a_far_date_is_uncertain():
    """C. A lone 'EXP' with only a far-away date must stay uncertain, not grab it."""
    detections = [
        {
            "text": "EXP",
            "confidence": 0.95,
            "box": [[250, 978], [300, 975], [301, 1002], [251, 1005]],
        },
        {
            # A valid date, but far away on the pack (different region) -> not near.
            "text": "17/JAN/27",
            "confidence": 0.97,
            "box": [[250, 300], [340, 300], [340, 326], [250, 326]],
        },
    ]

    candidates = extract_expiry_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.field_name == "EXPIRY"
    assert c.parsed_date is None, "must not associate a far-away date"
    assert c.uncertain is True
    return c


def test_multiple_nearby_dates_is_uncertain_not_guessed():
    """D. Two plausible nearby dates -> uncertain candidate, no guessing."""
    detections = [
        {
            "text": "EXP",
            "confidence": 0.96,
            "box": [[250, 978], [300, 978], [300, 1002], [250, 1002]],
        },
        {
            # to the right, same line
            "text": "17/JAN/27",
            "confidence": 0.96,
            "box": [[305, 978], [390, 978], [390, 1002], [305, 1002]],
        },
        {
            # just below the label, overlapping horizontally
            "text": "18/JAN/27",
            "confidence": 0.95,
            "box": [[250, 1006], [335, 1006], [335, 1030], [250, 1030]],
        },
    ]

    candidates = extract_expiry_candidates(detections)

    assert len(candidates) == 1, f"expected 1 candidate, got {len(candidates)}"
    c = candidates[0]
    assert c.field_name == "EXPIRY"
    assert c.parsed_date is None, "must not guess between two plausible dates"
    assert c.uncertain is True
    assert "multiple plausible nearby dates" in c.note, c.note
    return c


if __name__ == "__main__":
    results = [
        ("A same-detection", test_amul_exp_date_parsed()),
        ("B separate nearby", test_separate_nearby_detections_associated()),
        ("C far date -> uncertain", test_label_with_only_a_far_date_is_uncertain()),
        ("C' no date -> uncertain", test_unparseable_label_is_uncertain_not_invented()),
        ("D multiple -> uncertain", test_multiple_nearby_dates_is_uncertain_not_guessed()),
    ]
    for name, candidate in results:
        print(f"PASS: {name}")
        print("  candidate:", candidate.to_dict())
    print("\nAll tests passed.")
