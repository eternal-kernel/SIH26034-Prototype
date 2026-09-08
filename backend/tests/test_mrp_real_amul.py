"""Integration test: REAL OCR OUTPUT -> EXISTING MRP EXTRACTION -> CANDIDATE.

Feeds the actual detection produced by PaddleOCR on
``backend/amul_icecream.jpeg`` (``MRP:E260.00`` @ 0.9828) into the unmodified
``extract_mrp_candidates`` function. The parsed result is obtained by CALLING
the extractor — it is not hardcoded.

This exercises the real-world currency corruption case: the rupee sign was
misread as "E". The extractor must recover the numeric amount without ever
pretending "E" is ₹.

Runnable directly or via pytest.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.mrp import extract_mrp_candidates  # noqa: E402

# The real MRP detection from PaddleOCR on amul_icecream.jpeg.
REAL_MRP_BOX = [[603, 968], [886, 961], [887, 994], [604, 1001]]


def test_real_amul_mrp_detection():
    detections = [
        {
            "text": "MRP:E260.00",
            "confidence": 0.9828,
            "box": REAL_MRP_BOX,
            "source_image": "amul_icecream.jpeg",
        }
    ]

    candidates = extract_mrp_candidates(detections)

    assert len(candidates) == 1, f"expected exactly 1 candidate, got {len(candidates)}"
    c = candidates[0]

    assert c.field_name == "MRP", c.field_name
    assert c.raw_text == "MRP:E260.00", c.raw_text            # preserved verbatim
    assert c.parsed_value == 260.00, c.parsed_value           # numeric amount recovered
    assert c.currency is None, c.currency                     # 'E' NOT treated as ₹
    assert c.confidence == 0.9828, c.confidence
    assert c.source_image == "amul_icecream.jpeg", c.source_image
    assert c.box == REAL_MRP_BOX, c.box
    assert c.uncertain is False, c.note                       # value is clear
    return c


if __name__ == "__main__":
    candidate = test_real_amul_mrp_detection()
    print("PASS: test_real_amul_mrp_detection")
    print("  candidate:", candidate.to_dict())
    print("\nReal Amul MRP integration test passed.")
