"""Integration test: REAL OCR OUTPUT -> EXISTING QUANTITY EXTRACTION -> CANDIDATE.

Feeds the actual detections produced by PaddleOCR on
``backend/amul_icecream.jpeg`` into the unmodified
``extract_quantity_candidates`` function:

  - "Net Content:"  @ 0.9742
  - "1L/553g"        @ 0.9993   (a real multi-representation declaration)

The parsed result is obtained by CALLING the extractor — it is not hardcoded.
The candidate must preserve BOTH representations (1 L and 553 g).

Runnable directly or via pytest.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.quantity import extract_quantity_candidates  # noqa: E402

# Real bounding boxes from the OCR diagnostic run on amul_icecream.jpeg.
REAL_LABEL_BOX = [[105, 1055], [199, 1053], [200, 1075], [106, 1077]]
REAL_VALUE_BOX = [[223, 1051], [306, 1051], [306, 1075], [223, 1075]]


def test_real_amul_net_content():
    detections = [
        {
            "text": "Net Content:",
            "confidence": 0.9742,
            "box": REAL_LABEL_BOX,
            "source_image": "amul_icecream.jpeg",
        },
        {
            "text": "1L/553g",
            "confidence": 0.9993,
            "box": REAL_VALUE_BOX,
            "source_image": "amul_icecream.jpeg",
        },
    ]

    candidates = extract_quantity_candidates(detections)

    assert len(candidates) == 1, f"expected exactly 1 candidate, got {len(candidates)}"
    c = candidates[0]

    assert c.field_name == "NET_QUANTITY", c.field_name
    assert c.raw_text == "1L/553g", c.raw_text                 # preserved verbatim
    assert c.confidence == 0.9993, c.confidence
    assert c.source_image == "amul_icecream.jpeg", c.source_image
    assert c.box == REAL_VALUE_BOX, c.box
    assert c.uncertain is False, c.note

    # BOTH representations must be preserved, in order, neither dropped.
    reps = [(q["value"], q["unit"]) for q in c.quantities]
    assert reps == [(1.0, "L"), (553.0, "g")], reps

    # Label evidence retained from the separate label detection.
    assert c.label_text == "Net Content:", c.label_text
    return c


if __name__ == "__main__":
    candidate = test_real_amul_net_content()
    print("PASS: test_real_amul_net_content")
    print("  candidate:", candidate.to_dict())
    print("\nReal Amul net-quantity integration test passed.")
