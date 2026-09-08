"""Integration test: REAL OCR OUTPUT -> EXISTING ENTITY EXTRACTION -> CANDIDATE.

Feeds the actual detections produced by PaddleOCR on
``backend/amul_icecream.jpeg`` into the unmodified ``extract_entity_candidates``
function and asserts on what the extractor honestly produces.

Recorded from the diagnostic run, the only entity-type declaration on this
pack is a "Marketed by" line (detection index 14). There is NO explicit
Manufacturer / Packer / Importer declaration in the recorded OCR. This test
therefore verifies:

  - the "Marketed by" declaration is captured, with its role left UNKNOWN
    (we do not assert it is the legal manufacturer/packer/importer);
  - NO candidate claims the MANUFACTURER / PACKER / IMPORTER role.

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

from extraction.entity import extract_entity_candidates  # noqa: E402

# Real detections from the OCR diagnostic run on amul_icecream.jpeg.
# [14] the "Marketed by" declaration and [18] the address continuation line.
MARKETED_BY_TEXT = "Marketed by: Gujarat Co-operative Milk Marketing Federalion Lld., Amul"
MARKETED_BY_BOX = [[379, 682], [798, 676], [798, 696], [379, 702]]
ADDRESS_TEXT = "Dairy Road, Anand, Gujarat-388001, India. Website: www.amulicecream n"
ADDRESS_BOX = [[379, 700], [816, 693], [816, 710], [379, 717]]


def test_real_amul_entity_behavior():
    detections = [
        {"text": MARKETED_BY_TEXT, "confidence": 0.9630,
         "box": MARKETED_BY_BOX, "source_image": "amul_icecream.jpeg"},
        {"text": ADDRESS_TEXT, "confidence": 0.9457,
         "box": ADDRESS_BOX, "source_image": "amul_icecream.jpeg"},
    ]

    candidates = extract_entity_candidates(detections)

    # Exactly one entity-labelled declaration is present: the "Marketed by" line.
    assert len(candidates) == 1, f"expected 1 candidate, got {len(candidates)}"
    c = candidates[0]

    # It is preserved verbatim, with the label prefix separated for entity_text.
    assert c.field_name == "ENTITY"
    assert c.raw_text == MARKETED_BY_TEXT, c.raw_text
    assert c.entity_text == "Gujarat Co-operative Milk Marketing Federalion Lld., Amul", c.entity_text
    assert c.confidence == 0.9630
    assert c.source_image == "amul_icecream.jpeg"
    assert c.box == MARKETED_BY_BOX
    assert c.uncertain is False, c.note

    # HONEST FINDING: this is a "Marketed by" line -> role UNKNOWN, NOT a claimed
    # manufacturer/packer/importer.
    assert c.role == "UNKNOWN", c.role
    roles = {cand.role for cand in candidates}
    assert "MANUFACTURER" not in roles
    assert "PACKER" not in roles
    assert "IMPORTER" not in roles
    return c


if __name__ == "__main__":
    candidate = test_real_amul_entity_behavior()
    print("PASS: test_real_amul_entity_behavior")
    print("  candidate:", candidate.to_dict())
    print()
    print("HONEST FINDING: the real Amul OCR contains a 'Marketed by' declaration")
    print("(role=UNKNOWN). No explicit Manufacturer/Packer/Importer declaration")
    print("was present in the recorded OCR, so none was fabricated.")
