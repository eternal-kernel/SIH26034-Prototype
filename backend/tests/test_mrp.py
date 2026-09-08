"""Unit tests for extraction.mrp (synthetic OCR detections).

Runnable directly (``python tests/test_mrp.py`` from backend/) or via pytest.
"""

import os
import sys

# UTF-8 stdout so the ₹ symbol prints on the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# Make the backend/ directory importable regardless of how this file is run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.mrp import extract_mrp_candidates  # noqa: E402


def test_A_same_detection_with_rupee():
    """A. 'MRP:₹260.00' -> value 260.00, currency ₹, not uncertain."""
    detections = [
        {
            "text": "MRP:₹260.00",
            "confidence": 0.98,
            "box": [[600, 960], [890, 960], [890, 995], [600, 995]],
            "source_image": "synthetic.png",
        }
    ]
    candidates = extract_mrp_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.field_name == "MRP"
    assert c.raw_text == "MRP:₹260.00"
    assert c.parsed_value == 260.00, c.parsed_value
    assert c.currency == "₹", c.currency
    assert c.uncertain is False, c.note
    return c


def test_B_same_detection_no_currency():
    """B. 'MRP:260.00' -> value 260.00, currency unknown (None), not uncertain."""
    detections = [
        {
            "text": "MRP:260.00",
            "confidence": 0.97,
            "box": [[600, 960], [860, 960], [860, 995], [600, 995]],
        }
    ]
    candidates = extract_mrp_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.parsed_value == 260.00, c.parsed_value
    assert c.currency is None, c.currency          # must not invent a currency
    assert c.uncertain is False, c.note
    return c


def test_C_separate_detections_associated():
    """C. 'MRP' + nearby '₹260.00' -> associated conservatively by boxes."""
    detections = [
        {
            "text": "MRP",
            "confidence": 0.98,
            "box": [[600, 960], [660, 960], [660, 995], [600, 995]],
            "source_image": "synthetic.png",
        },
        {
            "text": "₹260.00",
            "confidence": 0.96,
            "box": [[665, 960], [880, 960], [880, 995], [665, 995]],
            "source_image": "synthetic.png",
        },
    ]
    candidates = extract_mrp_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.field_name == "MRP"
    assert c.raw_text == "₹260.00"                 # value comes from the value box
    assert c.parsed_value == 260.00, c.parsed_value
    assert c.currency == "₹", c.currency
    assert c.uncertain is False, c.note
    assert c.label_text == "MRP"
    assert c.box == [[665, 960], [880, 960], [880, 995], [665, 995]]
    return c


def test_D_currency_corruption_not_silently_fixed():
    """D. 'MRP:E260.00' -> value 260.00 but currency unknown; raw text intact."""
    detections = [
        {
            "text": "MRP:E260.00",
            "confidence": 0.9828,
            "box": [[603, 968], [886, 961], [887, 994], [604, 1001]],
            "source_image": "synthetic.png",
        }
    ]
    candidates = extract_mrp_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.raw_text == "MRP:E260.00", "raw OCR text must be preserved exactly"
    assert c.parsed_value == 260.00, c.parsed_value
    assert c.currency is None, "must NOT pretend 'E' is ₹"
    assert c.currency != "₹"
    assert "E" in c.note and "not interpreting it as" in c.note.lower()
    # value is clear even though the currency is suspicious
    assert c.uncertain is False, c.note
    return c


def test_E_label_with_no_usable_price_is_uncertain():
    """E. An MRP label with no numeric value -> uncertain, parsed_value None."""
    detections = [
        {
            "text": "MRP (incl. of all taxes):",
            "confidence": 0.95,
            "box": [[600, 960], [900, 960], [900, 995], [600, 995]],
        }
    ]
    candidates = extract_mrp_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.parsed_value is None
    assert c.uncertain is True
    return c


def test_F_multiple_nearby_prices_is_uncertain():
    """F. Two plausible nearby prices -> uncertain, not guessed."""
    detections = [
        {
            "text": "MRP",
            "confidence": 0.98,
            "box": [[600, 960], [660, 960], [660, 995], [600, 995]],
        },
        {
            "text": "₹260.00",
            "confidence": 0.96,
            "box": [[665, 960], [820, 960], [820, 995], [665, 995]],
        },
        {
            "text": "₹270.00",
            "confidence": 0.95,
            "box": [[600, 999], [755, 999], [755, 1030], [600, 1030]],
        },
    ]
    candidates = extract_mrp_candidates(detections)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.parsed_value is None, "must not guess between two prices"
    assert c.uncertain is True
    assert "multiple plausible nearby price values" in c.note, c.note
    return c


if __name__ == "__main__":
    results = [
        ("A same-detection ₹", test_A_same_detection_with_rupee()),
        ("B same-detection no-currency", test_B_same_detection_no_currency()),
        ("C separate nearby", test_C_separate_detections_associated()),
        ("D currency corruption", test_D_currency_corruption_not_silently_fixed()),
        ("E label no price", test_E_label_with_no_usable_price_is_uncertain()),
        ("F multiple nearby", test_F_multiple_nearby_prices_is_uncertain()),
    ]
    for name, candidate in results:
        print(f"PASS: {name}")
        print("  candidate:", candidate.to_dict())
    print("\nAll MRP unit tests passed.")
