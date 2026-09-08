"""Unit tests for extraction.entity (synthetic OCR detections).

Runnable directly (``python tests/test_entity.py`` from backend/) or via pytest.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.entity import extract_entity_candidates  # noqa: E402


def test_A_same_detection_manufacturer():
    """A. 'Manufactured by ABC Foods Pvt. Ltd.' -> MANUFACTURER."""
    detections = [
        {"text": "Manufactured by ABC Foods Pvt. Ltd.", "confidence": 0.98,
         "box": [[40, 100], [500, 100], [500, 130], [40, 130]]}
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.role == "MANUFACTURER", c.role
    assert c.raw_text == "Manufactured by ABC Foods Pvt. Ltd."
    assert c.entity_text == "ABC Foods Pvt. Ltd.", c.entity_text
    assert c.uncertain is False, c.note
    return c


def test_B_same_detection_packer():
    """B. 'Packed by XYZ Foods Ltd.' -> PACKER."""
    detections = [
        {"text": "Packed by XYZ Foods Ltd.", "confidence": 0.97,
         "box": [[40, 100], [420, 100], [420, 130], [40, 130]]}
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.role == "PACKER", c.role
    assert c.entity_text == "XYZ Foods Ltd.", c.entity_text
    assert c.uncertain is False, c.note
    return c


def test_C_same_detection_importer():
    """C. 'Imported by Global Foods Ltd.' -> IMPORTER."""
    detections = [
        {"text": "Imported by Global Foods Ltd.", "confidence": 0.96,
         "box": [[40, 100], [460, 100], [460, 130], [40, 130]]}
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.role == "IMPORTER", c.role
    assert c.entity_text == "Global Foods Ltd.", c.entity_text
    assert c.uncertain is False, c.note
    return c


def test_D_separate_detections_associated():
    """D. 'Manufactured by' + nearby 'ABC Foods Pvt. Ltd.' -> associated."""
    detections = [
        {"text": "Manufactured by", "confidence": 0.98,
         "box": [[40, 100], [200, 100], [200, 124], [40, 124]]},
        {"text": "ABC Foods Pvt. Ltd.", "confidence": 0.96,
         "box": [[210, 100], [470, 100], [470, 124], [210, 124]]},
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.role == "MANUFACTURER", c.role
    assert c.raw_text == "ABC Foods Pvt. Ltd."
    assert c.entity_text == "ABC Foods Pvt. Ltd."
    assert c.label_text.lower().startswith("manufactured by"), c.label_text
    assert c.box == [[210, 100], [470, 100], [470, 124], [210, 124]]
    assert c.uncertain is False, c.note
    return c


def test_E_far_away_entity_not_associated():
    """E. 'Manufactured by' + a distant company -> uncertain."""
    detections = [
        {"text": "Manufactured by", "confidence": 0.98,
         "box": [[40, 100], [200, 100], [200, 124], [40, 124]]},
        {"text": "ABC Foods Pvt. Ltd.", "confidence": 0.96,
         "box": [[40, 600], [300, 600], [300, 624], [40, 624]]},  # far below
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.entity_text is None
    assert c.uncertain is True
    return c


def test_F_multiple_nearby_is_uncertain():
    """F. Two plausible nearby company names -> uncertain, no guessing."""
    detections = [
        {"text": "Manufactured by", "confidence": 0.98,
         "box": [[40, 100], [200, 100], [200, 124], [40, 124]]},
        {"text": "ABC Foods Pvt. Ltd.", "confidence": 0.96,
         "box": [[210, 100], [430, 100], [430, 124], [210, 124]]},
        {"text": "DEF Foods Ltd.", "confidence": 0.95,
         "box": [[40, 128], [260, 128], [260, 152], [40, 152]]},
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.entity_text is None
    assert c.uncertain is True
    assert "multiple plausible nearby entity values" in c.note, c.note
    return c


def test_G_label_without_value_is_uncertain():
    """G. A bare label with no value -> uncertain."""
    detections = [
        {"text": "Manufacturer:", "confidence": 0.95,
         "box": [[40, 100], [230, 100], [230, 130], [40, 130]]}
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.entity_text is None
    assert c.uncertain is True
    return c


def test_H_garbled_entity_preserved_not_corrected():
    """H. Garbled entity name -> preserved verbatim, not invented/corrected."""
    detections = [
        {"text": "Manufactured by A8C F00ds Pvt Ltd", "confidence": 0.80,
         "box": [[40, 100], [500, 100], [500, 130], [40, 130]]}
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.role == "MANUFACTURER"
    assert c.raw_text == "Manufactured by A8C F00ds Pvt Ltd"     # exact
    assert c.entity_text == "A8C F00ds Pvt Ltd"                  # preserved, not corrected
    assert c.uncertain is False, c.note
    return c


def test_I_manufactured_and_marketed_by_preserved():
    """I. 'Manufactured & Marketed by ...' -> MANUFACTURER, info not lost."""
    detections = [
        {"text": "Manufactured & Marketed by ABC Foods Pvt. Ltd.", "confidence": 0.97,
         "box": [[40, 100], [600, 100], [600, 130], [40, 130]]}
    ]
    candidates = extract_entity_candidates(detections)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.role == "MANUFACTURER", c.role
    assert c.raw_text == "Manufactured & Marketed by ABC Foods Pvt. Ltd."
    assert c.entity_text == "ABC Foods Pvt. Ltd.", c.entity_text
    assert c.uncertain is False, c.note
    return c


if __name__ == "__main__":
    results = [
        ("A same-detection MANUFACTURER", test_A_same_detection_manufacturer()),
        ("B same-detection PACKER", test_B_same_detection_packer()),
        ("C same-detection IMPORTER", test_C_same_detection_importer()),
        ("D separate nearby", test_D_separate_detections_associated()),
        ("E far-away entity", test_E_far_away_entity_not_associated()),
        ("F multiple nearby", test_F_multiple_nearby_is_uncertain()),
        ("G label no value", test_G_label_without_value_is_uncertain()),
        ("H garbled preserved", test_H_garbled_entity_preserved_not_corrected()),
        ("I manufactured & marketed", test_I_manufactured_and_marketed_by_preserved()),
    ]
    for name, candidate in results:
        print(f"PASS: {name}")
        print("  candidate:", candidate.to_dict())
    print("\nAll entity unit tests passed.")
