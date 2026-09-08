"""Integration test: REAL OCR OUTPUT -> EXISTING EXTRACTION MODULE -> CANDIDATE.

Unlike the synthetic unit tests in ``test_expiry.py``, this test feeds the
actual detection produced by PaddleOCR on ``backend/amul_icecream.jpeg`` (as
recorded in the diagnostic run) into the unmodified ``extract_expiry_candidates``
function and asserts on the structured candidate it returns.

The parsed result is NOT hardcoded — it is obtained by calling the extractor.

Runnable directly (``python tests/test_expiry_real_amul.py`` from backend/) or
via pytest.
"""

import os
import sys

# Make the backend/ directory importable regardless of how this file is run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.expiry import extract_expiry_candidates  # noqa: E402

# The real EXP detection from PaddleOCR on amul_icecream.jpeg, in the same
# shape the diagnostic script (ocr_test.py) emits: text / confidence / box.
REAL_EXP_BOX = [[250, 978], [542, 971], [543, 1002], [251, 1009]]


def test_real_amul_exp_detection():
    detections = [
        {
            "text": "EXP:17/JAN/27",
            "confidence": 0.9631,
            "box": REAL_EXP_BOX,
            "source_image": "amul_icecream.jpeg",
        }
    ]

    candidates = extract_expiry_candidates(detections)

    assert len(candidates) == 1, f"expected exactly 1 candidate, got {len(candidates)}"
    c = candidates[0]

    assert c.field_name == "EXPIRY", c.field_name                      # (1)
    assert c.raw_text == "EXP:17/JAN/27", c.raw_text                   # (2)
    assert c.parsed_date == "2027-01-17", c.parsed_date                # (3)
    assert c.confidence == 0.9631, c.confidence                        # (4)
    assert c.source_image == "amul_icecream.jpeg", c.source_image      # (5)
    assert c.uncertain is False, c.note                                # (6)
    assert c.box == REAL_EXP_BOX, c.box                                # (7) box preserved
    return c


if __name__ == "__main__":
    candidate = test_real_amul_exp_detection()
    print("PASS: test_real_amul_exp_detection")
    print("  candidate:", candidate.to_dict())
    print("\nReal Amul integration test passed.")
