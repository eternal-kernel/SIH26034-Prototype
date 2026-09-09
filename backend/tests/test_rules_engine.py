"""Unit tests for rules_engine.py (the v1 deterministic Rules Engine).

Uses hand-built ``pipeline.PipelineResult``/``pipeline.OcrStatus`` instances
(mocked extraction output) -- no PaddleOCR, no real pipeline call, no image
decoding. Every test constructs the exact declarations dict shape the real
extraction modules already produce (verified against extraction/*.py and
pipeline.py earlier in this project) and feeds it straight to
``rules_engine.evaluate()``.

Runnable directly (``python tests/test_rules_engine.py`` from backend/) or
via pytest.
"""

import os
import sys
from datetime import date

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ImageQualityResult, OcrStatus, PipelineResult  # noqa: E402
from rules_engine import (  # noqa: E402
    NOT_ASSESSABLE,
    PASS,
    POTENTIAL_ISSUE,
    InspectionContext,
    RuleFinding,
    evaluate,
)

_ALL_EMPTY_DECLARATIONS = {
    "manufacturer_packer_importer": [],
    "generic_common_name": [],
    "net_quantity": [],
    "mrp": [],
    "best_before_use_by": [],
}


def _usable_quality(source_image=None):
    return ImageQualityResult(
        usable=True, issues=[], metrics={}, thresholds={}, source_image=source_image, note=""
    )


def _blocked_quality(source_image=None):
    return ImageQualityResult(
        usable=False, issues=["TOO_SMALL"], metrics={}, thresholds={}, source_image=source_image,
        note="image is likely unusable for OCR (retake suggested): TOO_SMALL",
    )


def _pipeline_result(declarations, source_image="test.jpg", detection_count=1):
    """A successful-OCR PipelineResult with the given declarations dict."""
    merged = dict(_ALL_EMPTY_DECLARATIONS)
    merged.update(declarations)
    return PipelineResult(
        source_image=source_image,
        image_quality=_usable_quality(source_image),
        ocr=OcrStatus(ran=True, detection_count=detection_count, note=""),
        detections=[{"text": "placeholder", "confidence": 0.9, "box": None, "source_image": source_image}],
        declarations=merged,
    )


def _blocked_pipeline_result(source_image="tiny.jpg"):
    return PipelineResult(
        source_image=source_image,
        image_quality=_blocked_quality(source_image),
        ocr=OcrStatus(ran=False, reason="image failed the hard image-quality gate"),
        detections=None,
        declarations=None,
    )


def _failed_ocr_pipeline_result(source_image="bad.jpg"):
    return PipelineResult(
        source_image=source_image,
        image_quality=_usable_quality(source_image),
        ocr=OcrStatus(ran=False, error="boom: paddle exploded"),
        detections=None,
        declarations=None,
    )


def _zero_detection_pipeline_result(source_image="blank.jpg"):
    return _pipeline_result(_ALL_EMPTY_DECLARATIONS, source_image=source_image, detection_count=0)


def _find(findings, rule_id) -> RuleFinding:
    for f in findings:
        if f.rule_id == rule_id:
            return f
    raise AssertionError(f"no finding for {rule_id}")


def _entity_candidate(role, entity_text, raw_text=None, uncertain=False, confidence=0.95, source_image="test.jpg"):
    return {
        "field_name": "ENTITY", "role": role, "raw_text": raw_text or entity_text,
        "entity_text": entity_text, "confidence": confidence, "box": [[0, 0], [1, 1]],
        "source_image": source_image, "uncertain": uncertain, "note": "",
        "label_text": None, "label_box": None,
    }


def _mrp_candidate(parsed_value, currency=None, raw_text=None, uncertain=False, confidence=0.9, source_image="test.jpg"):
    return {
        "field_name": "MRP", "raw_text": raw_text or f"MRP:{parsed_value}", "parsed_value": parsed_value,
        "currency": currency, "confidence": confidence, "box": [[0, 0], [1, 1]],
        "source_image": source_image, "uncertain": uncertain, "note": "",
        "label_text": None, "label_box": None,
    }


def _quantity_candidate(quantities, raw_text="Net Wt. 500 g", uncertain=False, confidence=0.95, source_image="test.jpg"):
    return {
        "field_name": "NET_QUANTITY", "raw_text": raw_text, "quantities": quantities,
        "confidence": confidence, "box": [[0, 0], [1, 1]], "source_image": source_image,
        "uncertain": uncertain, "note": "", "label_text": None, "label_box": None,
    }


def _date_candidate(field_name, parsed_date, raw_text=None, uncertain=False, confidence=0.95, source_image="test.jpg"):
    return {
        "field_name": field_name, "raw_text": raw_text or f"{field_name}:{parsed_date}",
        "parsed_date": parsed_date, "confidence": confidence, "box": [[0, 0], [1, 1]],
        "source_image": source_image, "uncertain": uncertain, "note": "",
        "label_text": None, "label_box": None,
    }


def _name_candidate(product_name, raw_text=None, uncertain=False, confidence=0.9, source_image="test.jpg"):
    return {
        "field_name": "PRODUCT_NAME", "raw_text": raw_text or f"Common Name: {product_name}",
        "product_name": product_name, "confidence": confidence, "box": [[0, 0], [1, 1]],
        "source_image": source_image, "uncertain": uncertain, "note": "",
        "label_text": None, "label_box": None,
    }


# ---------------------------------------------------------------------------
# Cross-cutting: OCR-failure / image-quality-blocked / zero-detection states
# must stay distinguishable across every rule.
# ---------------------------------------------------------------------------
def test_A_image_quality_blocked_is_not_assessable_for_every_rule():
    result = evaluate(_blocked_pipeline_result())
    assert len(result.findings) == 9
    for f in result.findings:
        assert f.result == NOT_ASSESSABLE, (f.rule_id, f.result)
        assert f.uncertainty_status.ocr_ran is False
        assert "image failed the hard image-quality gate" in f.reason
        assert f.evidence == []
    return result


def test_B_ocr_failure_is_not_assessable_and_distinct_from_quality_block():
    result = evaluate(_failed_ocr_pipeline_result())
    for f in result.findings:
        assert f.result == NOT_ASSESSABLE, (f.rule_id, f.result)
        assert "boom: paddle exploded" in f.reason
        # Must not be phrased as "declaration missing" -- must say OCR failed.
        assert "OCR failed" in f.reason
        assert "image failed the hard image-quality gate" not in f.reason
    return result


def test_C_zero_detections_is_potential_issue_not_not_assessable():
    """OCR ran successfully and found nothing -- this is real (empty)
    evidence, not a blocked/failed state, so most rules read POTENTIAL_ISSUE
    (a genuine absence), never NOT_ASSESSABLE."""
    result = evaluate(_zero_detection_pipeline_result())
    ent01 = _find(result.findings, "LM-ENT-01")
    assert ent01.result == POTENTIAL_ISSUE
    assert ent01.uncertainty_status.ocr_ran is True
    mrp01 = _find(result.findings, "LM-MRP-01")
    assert mrp01.result == POTENTIAL_ISSUE
    return result


def test_D_deterministic_same_input_same_output():
    pr = _pipeline_result({
        "manufacturer_packer_importer": [_entity_candidate("MANUFACTURER", "ABC Foods Pvt. Ltd.")],
    })
    ctx = InspectionContext(inspection_date=date(2026, 6, 1))
    result_1 = evaluate(pr, ctx).to_dict()
    result_2 = evaluate(pr, ctx).to_dict()
    assert result_1 == result_2
    return True


def test_E_result_states_are_never_a_legal_verdict():
    """Every finding's result must be one of exactly three literal strings."""
    pr = _pipeline_result({
        "manufacturer_packer_importer": [_entity_candidate("UNKNOWN", "Some Co.", raw_text="Marketed by: Some Co.")],
        "mrp": [_mrp_candidate(260.0)],
    })
    result = evaluate(pr)
    allowed = {PASS, POTENTIAL_ISSUE, NOT_ASSESSABLE}
    for f in result.findings:
        assert f.result in allowed, f.result
        for forbidden in ("VIOLATION", "LEGAL", "ILLEGAL", "COMPLIANT", "NON_COMPLIANT"):
            assert forbidden not in f.result
    return result


# ---------------------------------------------------------------------------
# LM-ENT-01 / LM-ENT-02
# ---------------------------------------------------------------------------
def test_F_entity_pass_when_confirmed_role_present():
    pr = _pipeline_result({
        "manufacturer_packer_importer": [_entity_candidate("MANUFACTURER", "ABC Foods Pvt. Ltd.")],
    })
    result = evaluate(pr)
    ent01, ent02 = _find(result.findings, "LM-ENT-01"), _find(result.findings, "LM-ENT-02")
    assert ent01.result == PASS
    assert ent02.result == PASS
    assert ent01.evidence[0].source_image == "test.jpg"
    return (ent01, ent02)


def test_G_marketed_by_never_becomes_manufacturer():
    """The exact real-world case: role=UNKNOWN must stay POTENTIAL_ISSUE on
    LM-ENT-02, never silently promoted to a confirmed role."""
    pr = _pipeline_result({
        "manufacturer_packer_importer": [
            _entity_candidate(
                "UNKNOWN", "Gujarat Co-operative Milk Marketing Federation Ltd., Amul",
                raw_text="Marketed by: Gujarat Co-operative Milk Marketing Federation Ltd., Amul",
                confidence=0.963,
            )
        ],
    })
    result = evaluate(pr)
    ent01 = _find(result.findings, "LM-ENT-01")
    ent02 = _find(result.findings, "LM-ENT-02")
    assert ent01.result == PASS  # a declaration WAS found and confidently read
    assert ent02.result == POTENTIAL_ISSUE  # but its role is not confirmed
    assert "not by itself satisfy" in ent02.reason or "could not be confirmed" in ent02.reason
    assert ent02.evidence[0].extracted_value == "UNKNOWN"
    return ent02


def test_H_entity_absent_is_potential_issue():
    result = evaluate(_pipeline_result({}))
    assert _find(result.findings, "LM-ENT-01").result == POTENTIAL_ISSUE
    assert _find(result.findings, "LM-ENT-02").result == NOT_ASSESSABLE  # nothing to assess role of
    return True


def test_I_multiple_confirmed_roles_is_not_a_conflict():
    """Manufacturer AND Packer both present is the normal, compliant case,
    not an ambiguity to flag."""
    pr = _pipeline_result({
        "manufacturer_packer_importer": [
            _entity_candidate("MANUFACTURER", "ABC Foods Pvt. Ltd."),
            _entity_candidate("PACKER", "XYZ Packers Ltd."),
        ],
    })
    result = evaluate(pr)
    assert _find(result.findings, "LM-ENT-01").result == PASS
    assert _find(result.findings, "LM-ENT-02").result == PASS
    return True


# ---------------------------------------------------------------------------
# LM-NAME-01
# ---------------------------------------------------------------------------
def test_J_name_present_passes_absent_is_issue():
    result_present = evaluate(_pipeline_result({"generic_common_name": [_name_candidate("Ice Cream")]}))
    assert _find(result_present.findings, "LM-NAME-01").result == PASS

    result_absent = evaluate(_pipeline_result({}))
    assert _find(result_absent.findings, "LM-NAME-01").result == POTENTIAL_ISSUE
    return True


# ---------------------------------------------------------------------------
# LM-QTY-01 / LM-QTY-02
# ---------------------------------------------------------------------------
def test_K_quantity_1L_553g_single_candidate_multi_representation_passes():
    """'1L/553g' is ONE declaration with two representations -- must PASS
    both QTY-01 (parseable) and QTY-02 (not a conflict)."""
    pr = _pipeline_result({
        "net_quantity": [
            _quantity_candidate(
                quantities=[{"value": 1.0, "unit": "L", "raw": "1L"}, {"value": 553.0, "unit": "g", "raw": "553g"}],
                raw_text="1L/553g",
            )
        ],
    })
    result = evaluate(pr)
    qty01, qty02 = _find(result.findings, "LM-QTY-01"), _find(result.findings, "LM-QTY-02")
    assert qty01.result == PASS
    assert qty02.result == PASS
    assert qty02.evidence[0].raw_text == "1L/553g"
    return (qty01, qty02)


def test_L_quantity_two_separate_candidates_is_potential_issue():
    pr = _pipeline_result({
        "net_quantity": [
            _quantity_candidate(quantities=[{"value": 500.0, "unit": "g", "raw": "500g"}], raw_text="Net Wt. 500g"),
            _quantity_candidate(quantities=[{"value": 600.0, "unit": "g", "raw": "600g"}], raw_text="Net Wt. 600g"),
        ],
    })
    result = evaluate(pr)
    assert _find(result.findings, "LM-QTY-01").result == PASS  # at least one is parseable
    qty02 = _find(result.findings, "LM-QTY-02")
    assert qty02.result == POTENTIAL_ISSUE
    assert len(qty02.evidence) == 2
    return qty02


def test_M_quantity_label_with_no_parseable_value_is_potential_issue():
    pr = _pipeline_result({"net_quantity": [_quantity_candidate(quantities=[], raw_text="Net Wt. 50", uncertain=True)]})
    result = evaluate(pr)
    assert _find(result.findings, "LM-QTY-01").result == POTENTIAL_ISSUE
    return True


# ---------------------------------------------------------------------------
# LM-MRP-01
# ---------------------------------------------------------------------------
def test_N_mrp_corrupted_currency_character_still_passes_on_value():
    """The real Amul case: 'MRP:E260.00' -- currency unknown, but the
    numeric value is confidently parsed, so LM-MRP-01 must PASS."""
    pr = _pipeline_result({
        "mrp": [_mrp_candidate(260.0, currency=None, raw_text="MRP:E260.00", confidence=0.9828)],
    })
    result = evaluate(pr)
    mrp01 = _find(result.findings, "LM-MRP-01")
    assert mrp01.result == PASS
    assert mrp01.evidence[0].extracted_value == 260.0
    assert mrp01.evidence[0].raw_text == "MRP:E260.00"
    return mrp01


def test_O_mrp_absent_and_unparseable_and_conflicting():
    absent = evaluate(_pipeline_result({}))
    assert _find(absent.findings, "LM-MRP-01").result == POTENTIAL_ISSUE

    unparseable = evaluate(_pipeline_result({"mrp": [_mrp_candidate(None, raw_text="MRP (incl. of all taxes):", uncertain=True)]}))
    assert _find(unparseable.findings, "LM-MRP-01").result == POTENTIAL_ISSUE

    conflicting = evaluate(_pipeline_result({
        "mrp": [_mrp_candidate(260.0, raw_text="MRP 260.00"), _mrp_candidate(270.0, raw_text="MRP 270.00")],
    }))
    conflict_finding = _find(conflicting.findings, "LM-MRP-01")
    assert conflict_finding.result == POTENTIAL_ISSUE
    assert len(conflict_finding.evidence) == 2
    return True


def test_P_lm_mrp_02_does_not_exist():
    """LM-MRP-02 was intentionally removed by the spec's legal-source
    verification pass and must never be recreated."""
    pr = _pipeline_result({"mrp": [_mrp_candidate(260.0, currency=None)]})
    result = evaluate(pr)
    rule_ids = {f.rule_id for f in result.findings}
    assert "LM-MRP-02" not in rule_ids
    assert len(result.findings) == 9
    return rule_ids


# ---------------------------------------------------------------------------
# LM-DATE-01 / 02 / 03
# ---------------------------------------------------------------------------
def test_Q_date_future_expiry_passes_all_three_date_rules():
    pr = _pipeline_result({
        "best_before_use_by": [_date_candidate("EXPIRY", "2099-01-17", raw_text="EXP:17/JAN/99")],
    })
    ctx = InspectionContext(inspection_date=date(2026, 9, 9))
    result = evaluate(pr, ctx)
    assert _find(result.findings, "LM-DATE-01").result == PASS
    assert _find(result.findings, "LM-DATE-03").result == PASS
    assert _find(result.findings, "LM-DATE-02").result == PASS
    return result


def test_R_date_elapsed_expiry_is_potential_issue_not_violation():
    pr = _pipeline_result({
        "best_before_use_by": [_date_candidate("EXPIRY", "2020-01-17", raw_text="EXP:17/JAN/20")],
    })
    ctx = InspectionContext(inspection_date=date(2026, 9, 9))
    result = evaluate(pr, ctx)
    date02 = _find(result.findings, "LM-DATE-02")
    assert date02.result == POTENTIAL_ISSUE
    assert "elapsed" in date02.reason
    assert "use-by/expiry" in date02.reason
    for forbidden in ("VIOLATION", "expired", "illegal"):
        assert forbidden not in date02.reason
    return date02


def test_S_best_before_vs_use_by_wording_is_distinct():
    """BEST_BEFORE and EXPIRY/USE_BY elapsing must never be collapsed into
    identical wording."""
    best_before_pr = _pipeline_result({
        "best_before_use_by": [_date_candidate("BEST_BEFORE", "2020-01-01")],
    })
    expiry_pr = _pipeline_result({
        "best_before_use_by": [_date_candidate("EXPIRY", "2020-01-01")],
    })
    ctx = InspectionContext(inspection_date=date(2026, 9, 9))

    bb_finding = _find(evaluate(best_before_pr, ctx).findings, "LM-DATE-02")
    exp_finding = _find(evaluate(expiry_pr, ctx).findings, "LM-DATE-02")

    assert bb_finding.result == POTENTIAL_ISSUE
    assert exp_finding.result == POTENTIAL_ISSUE
    assert bb_finding.reason != exp_finding.reason
    assert "best-before" in bb_finding.reason
    assert "peak quality" in bb_finding.reason
    assert "use-by/expiry" in exp_finding.reason
    assert "best-before" not in exp_finding.reason
    return (bb_finding, exp_finding)


def test_T_date_elapsed_check_requires_explicit_inspection_date():
    """No inspection_date supplied -> NOT_ASSESSABLE, never a silently
    invented 'today'."""
    pr = _pipeline_result({
        "best_before_use_by": [_date_candidate("EXPIRY", "2020-01-01")],
    })
    result = evaluate(pr, InspectionContext())  # inspection_date left None
    date02 = _find(result.findings, "LM-DATE-02")
    assert date02.result == NOT_ASSESSABLE
    assert "no inspection date" in date02.reason.lower()
    return date02


def test_U_date_unparseable_is_potential_issue_not_invented():
    pr = _pipeline_result({
        "best_before_use_by": [_date_candidate("EXPIRY", None, raw_text="EXP: XX/XX/XX", uncertain=True)],
    })
    result = evaluate(pr, InspectionContext(inspection_date=date(2026, 9, 9)))
    date03 = _find(result.findings, "LM-DATE-03")
    date02 = _find(result.findings, "LM-DATE-02")
    assert date03.result == POTENTIAL_ISSUE
    assert date02.result == NOT_ASSESSABLE  # nothing parseable to compare
    return (date03, date02)


def test_V_date_absent_unknown_category_is_not_assessable():
    result = evaluate(_pipeline_result({}), InspectionContext(category=None))
    date01 = _find(result.findings, "LM-DATE-01")
    assert date01.result == NOT_ASSESSABLE
    assert date01.uncertainty_status.applicability_known is False
    return date01


def test_W_date_absent_food_category_is_potential_issue():
    result = evaluate(_pipeline_result({}), InspectionContext(category="food"))
    date01 = _find(result.findings, "LM-DATE-01")
    assert date01.result == POTENTIAL_ISSUE
    return date01


def test_X_date_absent_non_food_category_is_not_assessable():
    result = evaluate(_pipeline_result({}), InspectionContext(category="electronics"))
    date01 = _find(result.findings, "LM-DATE-01")
    assert date01.result == NOT_ASSESSABLE
    return date01


if __name__ == "__main__":
    checks = [
        ("A image-quality blocked -> NOT_ASSESSABLE for all 9", test_A_image_quality_blocked_is_not_assessable_for_every_rule),
        ("B OCR failure distinct from quality block", test_B_ocr_failure_is_not_assessable_and_distinct_from_quality_block),
        ("C zero detections -> POTENTIAL_ISSUE not NOT_ASSESSABLE", test_C_zero_detections_is_potential_issue_not_not_assessable),
        ("D deterministic: same input -> same output", test_D_deterministic_same_input_same_output),
        ("E result states never a legal verdict", test_E_result_states_are_never_a_legal_verdict),
        ("F entity PASS on confirmed role", test_F_entity_pass_when_confirmed_role_present),
        ("G Marketed-by never becomes manufacturer", test_G_marketed_by_never_becomes_manufacturer),
        ("H entity absent -> POTENTIAL_ISSUE / NOT_ASSESSABLE", test_H_entity_absent_is_potential_issue),
        ("I multiple confirmed roles is not a conflict", test_I_multiple_confirmed_roles_is_not_a_conflict),
        ("J name present/absent", test_J_name_present_passes_absent_is_issue),
        ("K quantity 1L/553g single-candidate passes", test_K_quantity_1L_553g_single_candidate_multi_representation_passes),
        ("L quantity two separate candidates -> conflict", test_L_quantity_two_separate_candidates_is_potential_issue),
        ("M quantity unparseable -> issue", test_M_quantity_label_with_no_parseable_value_is_potential_issue),
        ("N MRP corrupted currency still passes on value", test_N_mrp_corrupted_currency_character_still_passes_on_value),
        ("O MRP absent/unparseable/conflicting", test_O_mrp_absent_and_unparseable_and_conflicting),
        ("P LM-MRP-02 does not exist", test_P_lm_mrp_02_does_not_exist),
        ("Q future expiry passes all three date rules", test_Q_date_future_expiry_passes_all_three_date_rules),
        ("R elapsed expiry is POTENTIAL_ISSUE not violation", test_R_date_elapsed_expiry_is_potential_issue_not_violation),
        ("S Best Before vs Use By wording distinct", test_S_best_before_vs_use_by_wording_is_distinct),
        ("T elapsed check requires explicit inspection date", test_T_date_elapsed_check_requires_explicit_inspection_date),
        ("U unparseable date -> issue, not invented", test_U_date_unparseable_is_potential_issue_not_invented),
        ("V date absent, unknown category -> NOT_ASSESSABLE", test_V_date_absent_unknown_category_is_not_assessable),
        ("W date absent, food category -> POTENTIAL_ISSUE", test_W_date_absent_food_category_is_potential_issue),
        ("X date absent, non-food category -> NOT_ASSESSABLE", test_X_date_absent_non_food_category_is_not_assessable),
    ]
    for name, fn in checks:
        fn()
        print(f"PASS: {name}")
    print("\nAll rules-engine tests passed.")
