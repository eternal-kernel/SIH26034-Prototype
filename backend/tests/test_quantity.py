"""Unit tests for extraction.quantity (synthetic OCR detections).

Runnable directly (``python tests/test_quantity.py`` from backend/) or via pytest.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.quantity import extract_quantity_candidates  # noqa: E402


def _units(quantities):
    return [(q["value"], q["unit"]) for q in quantities]


def test_A_same_detection_net_wt():
    """A. 'Net Wt. 500 g' -> one quantity 500 g, not uncertain."""
    detections = [
        {"text": "Net Wt. 500 g", "confidence": 0.98,
         "box": [[40, 100], [220, 100], [220, 130], [40, 130]]}
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.field_name == "NET_QUANTITY"
    assert _units(c.quantities) == [(500.0, "g")], c.quantities
    assert c.uncertain is False, c.note
    return c


def test_B_same_detection_net_quantity_litre():
    """B. 'Net Quantity: 1 L' -> one quantity 1 L, not uncertain."""
    detections = [
        {"text": "Net Quantity: 1 L", "confidence": 0.97,
         "box": [[40, 100], [260, 100], [260, 130], [40, 130]]}
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert _units(c.quantities) == [(1.0, "L")], c.quantities
    assert c.uncertain is False, c.note
    return c


def test_C_separate_detections_associated():
    """C. 'Net Content:' + nearby '500 ml' -> associated by boxes."""
    detections = [
        {"text": "Net Content:", "confidence": 0.97,
         "box": [[100, 100], [200, 100], [200, 124], [100, 124]]},
        {"text": "500 ml", "confidence": 0.96,
         "box": [[210, 100], [320, 100], [320, 124], [210, 124]]},
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.raw_text == "500 ml"
    assert _units(c.quantities) == [(500.0, "ml")], c.quantities
    assert c.uncertain is False, c.note
    assert c.label_text == "Net Content:"
    return c


def test_D_multiple_representations_both_preserved():
    """D. 'Net Content:' + nearby '1L/553g' -> BOTH 1 L and 553 g preserved."""
    detections = [
        {"text": "Net Content:", "confidence": 0.97,
         "box": [[105, 1053], [200, 1053], [200, 1077], [105, 1077]]},
        {"text": "1L/553g", "confidence": 0.99,
         "box": [[223, 1051], [306, 1051], [306, 1075], [223, 1075]]},
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.raw_text == "1L/553g"
    assert _units(c.quantities) == [(1.0, "L"), (553.0, "g")], c.quantities
    assert c.uncertain is False, c.note
    return c


def test_E_label_with_no_usable_quantity_is_uncertain():
    """E. A net label alone -> quantities=[], uncertain=True."""
    detections = [
        {"text": "Net Weight:", "confidence": 0.95,
         "box": [[40, 100], [200, 100], [200, 130], [40, 130]]}
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.quantities == []
    assert c.uncertain is True
    return c


def test_F_far_away_quantity_not_associated():
    """F. A far-away quantity must not be grabbed -> uncertain."""
    detections = [
        {"text": "Net Content:", "confidence": 0.97,
         "box": [[100, 100], [200, 100], [200, 124], [100, 124]]},
        {"text": "500 ml", "confidence": 0.96,
         "box": [[100, 600], [210, 600], [210, 624], [100, 624]]},  # far below
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.quantities == []
    assert c.uncertain is True
    return c


def test_G_multiple_nearby_quantities_is_uncertain():
    """G. Two plausible nearby quantity detections -> uncertain, not guessed."""
    detections = [
        {"text": "Net Content:", "confidence": 0.97,
         "box": [[100, 100], [200, 100], [200, 124], [100, 124]]},
        {"text": "500 g", "confidence": 0.96,
         "box": [[210, 100], [300, 100], [300, 124], [210, 124]]},
        {"text": "600 g", "confidence": 0.95,
         "box": [[100, 128], [190, 128], [190, 152], [100, 152]]},
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.quantities == []
    assert c.uncertain is True
    assert "multiple plausible nearby quantity detections" in c.note, c.note
    return c


def test_H_missing_unit_is_uncertain_not_invented():
    """H. 'Net Wt. 50' (unit missing) -> uncertain, no invented unit."""
    detections = [
        {"text": "Net Wt. 50", "confidence": 0.94,
         "box": [[40, 100], [180, 100], [180, 130], [40, 130]]}
    ]
    candidates = extract_quantity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.quantities == [], "must not invent a unit"
    assert c.uncertain is True
    assert "no recognizable unit" in c.note, c.note
    return c


if __name__ == "__main__":
    results = [
        ("A same-detection 500 g", test_A_same_detection_net_wt()),
        ("B same-detection 1 L", test_B_same_detection_net_quantity_litre()),
        ("C separate nearby 500 ml", test_C_separate_detections_associated()),
        ("D multiple reps 1L/553g", test_D_multiple_representations_both_preserved()),
        ("E label no quantity", test_E_label_with_no_usable_quantity_is_uncertain()),
        ("F far-away quantity", test_F_far_away_quantity_not_associated()),
        ("G multiple nearby", test_G_multiple_nearby_quantities_is_uncertain()),
        ("H missing unit", test_H_missing_unit_is_uncertain_not_invented()),
    ]
    for name, candidate in results:
        print(f"PASS: {name}")
        print("  candidate:", candidate.to_dict())
    print("\nAll quantity unit tests passed.")
