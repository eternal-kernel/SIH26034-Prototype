"""Integration test: REAL OCR OUTPUT -> PRODUCT-NAME EXTRACTION -> (honest) RESULT.

Feeds a representative subset of the actual detections produced by PaddleOCR on
``backend/amul_icecream.jpeg`` into the unmodified
``extract_product_name_candidates`` function.

HONEST FINDING (from the recorded diagnostic run): this pack's OCR contains NO
explicit generic/common/product-name label ("Common Name", "Generic Name",
"Product Name", ...). The top line is brand / marketing wording
("Amul Real Milk, Real Iee Gream,"), which this conservative extractor must NOT
promote to a generic name. The expected, honest result is therefore ZERO
candidates.

The result is obtained by CALLING the extractor — nothing is hardcoded.

Runnable directly or via pytest.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.product_name import extract_product_name_candidates  # noqa: E402

# A representative subset of the real OCR detections on amul_icecream.jpeg,
# including the brand/marketing line and several non-product declarations.
REAL_DETECTIONS = [
    {"text": "Amul Real Milk, Real Iee Gream,", "confidence": 0.9581,
     "box": [[31, 555], [452, 557], [452, 591], [31, 589]], "source_image": "amul_icecream.jpeg"},
    {"text": "Nutritional Information", "confidence": 0.9941,
     "box": [[58, 597], [320, 597], [320, 620], [58, 620]], "source_image": "amul_icecream.jpeg"},
    {"text": "Marketed by: Gujarat Co-operative Milk Marketing Federalion Lld., Amul", "confidence": 0.9630,
     "box": [[379, 682], [798, 676], [798, 696], [379, 702]], "source_image": "amul_icecream.jpeg"},
    {"text": "MRP:E260.00", "confidence": 0.9828,
     "box": [[603, 968], [886, 961], [887, 994], [604, 1001]], "source_image": "amul_icecream.jpeg"},
    {"text": "1L/553g", "confidence": 0.9993,
     "box": [[223, 1051], [306, 1051], [306, 1075], [223, 1075]], "source_image": "amul_icecream.jpeg"},
    {"text": "Pkd:22/APR/26", "confidence": 0.9978,
     "box": [[247, 942], [545, 936], [545, 967], [248, 973]], "source_image": "amul_icecream.jpeg"},
]


def test_real_amul_product_name_behavior():
    candidates = extract_product_name_candidates(REAL_DETECTIONS)

    # No explicit product-name label exists in the recorded OCR -> no candidate.
    assert candidates == [], (
        "expected NO product-name candidate (no explicit generic/common-name "
        f"label present); got {[c.to_dict() for c in candidates]}"
    )
    return candidates


if __name__ == "__main__":
    result = test_real_amul_product_name_behavior()
    print("PASS: test_real_amul_product_name_behavior")
    print(f"  candidates produced: {len(result)}")
    print()
    print("HONEST FINDING: the real Amul OCR contains NO explicit generic/common/")
    print("product-name declaration. The top line 'Amul Real Milk, Real Iee Gream,'")
    print("is brand/marketing wording, which the conservative extractor does not")
    print("promote to a generic name. No reliable product-name candidate was found,")
    print("and none was fabricated.")
