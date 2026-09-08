"""Unit tests for extraction.product_name (synthetic OCR detections).

Runnable directly (``python tests/test_product_name.py`` from backend/) or via pytest.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.product_name import extract_product_name_candidates  # noqa: E402


def test_A_same_detection_common_name():
    """A. 'Common Name: Chocolate Cream Biscuits' -> PRODUCT_NAME."""
    detections = [
        {"text": "Common Name: Chocolate Cream Biscuits", "confidence": 0.98,
         "box": [[40, 100], [520, 100], [520, 130], [40, 130]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.field_name == "PRODUCT_NAME"
    assert c.raw_text == "Common Name: Chocolate Cream Biscuits"
    assert c.product_name == "Chocolate Cream Biscuits", c.product_name
    assert c.uncertain is False, c.note
    return c


def test_B_same_detection_generic_name():
    """B. 'Generic Name: Vanilla Flavoured Ice Cream' -> PRODUCT_NAME."""
    detections = [
        {"text": "Generic Name: Vanilla Flavoured Ice Cream", "confidence": 0.97,
         "box": [[40, 100], [560, 100], [560, 130], [40, 130]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.product_name == "Vanilla Flavoured Ice Cream", c.product_name
    assert c.uncertain is False, c.note
    return c


def test_C_separate_detections_associated():
    """C. 'Common Name:' + nearby 'Chocolate Cream Biscuits' -> associated."""
    detections = [
        {"text": "Common Name:", "confidence": 0.97,
         "box": [[40, 100], [220, 100], [220, 124], [40, 124]]},
        {"text": "Chocolate Cream Biscuits", "confidence": 0.96,
         "box": [[230, 100], [560, 100], [560, 124], [230, 124]]},
    ]
    candidates = extract_product_name_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.raw_text == "Chocolate Cream Biscuits"
    assert c.product_name == "Chocolate Cream Biscuits"
    assert c.label_text.lower().startswith("common name")
    assert c.box == [[230, 100], [560, 100], [560, 124], [230, 124]]
    assert c.uncertain is False, c.note
    return c


def test_D_label_without_value_is_uncertain():
    """D. 'Product Name:' with no nearby value -> uncertain."""
    detections = [
        {"text": "Product Name:", "confidence": 0.95,
         "box": [[40, 100], [240, 100], [240, 130], [40, 130]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.product_name is None
    assert c.uncertain is True
    return c


def test_E_multiple_nearby_is_uncertain():
    """E. 'Common Name:' + two plausible nearby values -> uncertain, no guess."""
    detections = [
        {"text": "Common Name:", "confidence": 0.97,
         "box": [[40, 100], [220, 100], [220, 124], [40, 124]]},
        {"text": "Chocolate Cream Biscuits", "confidence": 0.96,
         "box": [[230, 100], [520, 100], [520, 124], [230, 124]]},
        {"text": "Vanilla Ice Cream", "confidence": 0.95,
         "box": [[40, 128], [300, 128], [300, 152], [40, 152]]},
    ]
    candidates = extract_product_name_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.product_name is None
    assert c.uncertain is True
    assert "multiple plausible nearby product-name values" in c.note, c.note
    return c


def test_F_far_away_not_associated():
    """F. Far-away product-looking text -> not associated; uncertain."""
    detections = [
        {"text": "Common Name:", "confidence": 0.97,
         "box": [[40, 100], [220, 100], [220, 124], [40, 124]]},
        {"text": "Chocolate Cream Biscuits", "confidence": 0.96,
         "box": [[40, 600], [360, 600], [360, 624], [40, 624]]},  # far below
    ]
    candidates = extract_product_name_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.product_name is None
    assert c.uncertain is True
    return c


def test_G_brand_only_not_classified():
    """G. 'AMUL' alone (no label) -> not treated as a product name."""
    detections = [
        {"text": "AMUL", "confidence": 0.99,
         "box": [[40, 100], [160, 100], [160, 140], [40, 140]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert candidates == [], "brand-only text must not be classified as product name"
    return None


def test_H_garbled_value_preserved():
    """H. Garbled 'Common Name: Vanila Flavoured Ice Creem' -> preserved verbatim."""
    detections = [
        {"text": "Common Name: Vanila Flavoured Ice Creem", "confidence": 0.80,
         "box": [[40, 100], [560, 100], [560, 130], [40, 130]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.raw_text == "Common Name: Vanila Flavoured Ice Creem"     # exact
    assert c.product_name == "Vanila Flavoured Ice Creem"             # not corrected
    assert c.uncertain is False, c.note
    return c


def test_I_mrp_not_product_name():
    """I. 'MRP: ₹260.00' -> not extracted as a product name."""
    detections = [
        {"text": "MRP: ₹260.00", "confidence": 0.98,
         "box": [[40, 100], [300, 100], [300, 130], [40, 130]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert candidates == []
    return None


def test_J_manufacturer_not_product_name():
    """J. 'Manufactured by ABC Foods Pvt. Ltd.' -> not a product name."""
    detections = [
        {"text": "Manufactured by ABC Foods Pvt. Ltd.", "confidence": 0.98,
         "box": [[40, 100], [500, 100], [500, 130], [40, 130]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert candidates == []
    return None


def test_K_quantity_not_product_name():
    """K. 'Net Content: 1L/553g' -> not a product name."""
    detections = [
        {"text": "Net Content: 1L/553g", "confidence": 0.98,
         "box": [[40, 100], [340, 100], [340, 130], [40, 130]]}
    ]
    candidates = extract_product_name_candidates(detections)
    assert candidates == []
    return None


if __name__ == "__main__":
    results = [
        ("A same-detection common name", test_A_same_detection_common_name()),
        ("B generic name", test_B_same_detection_generic_name()),
        ("C separate nearby", test_C_separate_detections_associated()),
        ("D label no value", test_D_label_without_value_is_uncertain()),
        ("E multiple nearby", test_E_multiple_nearby_is_uncertain()),
        ("F far-away", test_F_far_away_not_associated()),
        ("H garbled preserved", test_H_garbled_value_preserved()),
    ]
    for name, candidate in results:
        print(f"PASS: {name}")
        print("  candidate:", candidate.to_dict())

    # Rejection cases (no candidate expected).
    for name, fn in [
        ("G brand-only rejected", test_G_brand_only_not_classified),
        ("I MRP rejected", test_I_mrp_not_product_name),
        ("J manufacturer rejected", test_J_manufacturer_not_product_name),
        ("K quantity rejected", test_K_quantity_not_product_name),
    ]:
        fn()
        print(f"PASS: {name} (no candidate produced)")

    print("\nAll product-name unit tests passed.")
