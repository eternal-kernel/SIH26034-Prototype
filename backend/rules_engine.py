"""Deterministic Rules Engine (v1) for the SIH26034 backend.

Implements the nine executable v1 rules defined in
``docs/rules-engine-spec.md`` §7 (LM-ENT-01/02, LM-NAME-01, LM-QTY-01/02,
LM-MRP-01, LM-DATE-01/02/03). LM-MRP-02 was intentionally removed by that
specification's legal-source verification pass and is deliberately NOT
implemented here — see §17A.2 / §20 of the spec for why.

This module consumes an existing ``pipeline.PipelineResult`` unchanged. It
does **not** run OCR, does **not** call the pipeline, and does **not**
modify anything upstream — it is a pure, deterministic downstream reader.

VERY IMPORTANT — SCOPE (binding on every rule below):
  This module performs NO legal/compliance reasoning. It NEVER produces a
  verdict of "violation", "illegal", "compliant", or "non-compliant" — only
  one of exactly three result states (PASS / POTENTIAL_ISSUE /
  NOT_ASSESSABLE), each carrying the evidence it was based on. A human Legal
  Metrology inspector makes the actual compliance determination; this module
  produces preliminary findings for that inspector to review, nothing more.

Determinism: the same ``PipelineResult`` + the same ``InspectionContext``
always produces the same findings. No randomness, no wall-clock reads (the
inspection date must be supplied explicitly via ``InspectionContext`` — see
``evaluate()``'s docstring), no network, no LLM.

Core principle: AI/OCR READS. RULES EVALUATE. EVIDENCE SUPPORTS.
INSPECTOR DECIDES.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as date_type
from typing import Any, Dict, List, Optional

from pipeline import PipelineResult

# --------------------------------------------------------------------------
# Versioning (see docs/rules-engine-spec.md §16)
# --------------------------------------------------------------------------
RULESET_VERSION = "1.0.0"

# Every rule's own version. Bump per-rule (not the whole ruleset) for a
# change scoped to that rule only; see the spec's PATCH/MINOR/MAJOR policy.
_RULE_VERSIONS: Dict[str, str] = {
    "LM-ENT-01": "1.0.0",
    "LM-ENT-02": "1.0.0",
    "LM-NAME-01": "1.0.0",
    "LM-QTY-01": "1.0.0",
    "LM-QTY-02": "1.0.0",
    "LM-MRP-01": "1.0.0",
    "LM-DATE-01": "1.0.0",
    "LM-DATE-02": "1.0.0",
    "LM-DATE-03": "1.0.0",
}

# --------------------------------------------------------------------------
# Result states — EXACTLY these three. Never add a fourth. Never rename one
# of these to (or otherwise produce) "VIOLATION"/"LEGAL"/"ILLEGAL"/
# "COMPLIANT"/"NON_COMPLIANT" or any equivalent legal conclusion.
# --------------------------------------------------------------------------
PASS = "PASS"
POTENTIAL_ISSUE = "POTENTIAL_ISSUE"
NOT_ASSESSABLE = "NOT_ASSESSABLE"

_VALID_RESULTS = {PASS, POTENTIAL_ISSUE, NOT_ASSESSABLE}


# --------------------------------------------------------------------------
# Typed structures (spec §6.2, §8)
# --------------------------------------------------------------------------
@dataclass
class InspectionContext:
    """Caller-supplied inspection context (spec §6.2).

    Every field defaults to ``None`` (unknown). Nothing in this module ever
    infers ``category`` or ``known_exemption`` from OCR text or the image —
    they must be supplied by the caller (inspector, form, or test).

    ``inspection_date`` is required for LM-DATE-02's elapsed-date
    comparison. It is deliberately NOT defaulted to "today": a rules engine
    that silently reads the wall clock would make the same finding
    non-reproducible for the same stored ``PipelineResult`` depending on
    when it happens to be re-evaluated. If it is ``None``, LM-DATE-02
    returns NOT_ASSESSABLE rather than guessing "now".
    """

    category: Optional[str] = None
    inspection_type: Optional[str] = None
    is_prepackaged: Optional[bool] = None
    intended_for_retail_sale: Optional[bool] = None
    known_exemption: Optional[str] = None
    inspection_date: Optional[date_type] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["inspection_date"] = self.inspection_date.isoformat() if self.inspection_date else None
        return d


@dataclass
class Evidence:
    """One piece of evidence backing a finding — copied verbatim from an
    extraction candidate dict. Never fabricated: every field here comes
    directly from a real candidate already produced by ``extraction.*``."""

    source_image: Optional[str]
    raw_text: Optional[str]
    extracted_value: Any
    confidence: Optional[float]
    box: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyStatus:
    """The two orthogonal uncertainty axes (spec §9): did OCR run at all,
    and is the candidate evidence itself marked uncertain, and is rule
    applicability known. None of these, at any level, upgrades a finding
    into a legal conclusion."""

    ocr_ran: bool
    candidate_uncertain: Optional[bool]
    applicability_known: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleFinding:
    """One rule's finding for one image/inspection (spec §8)."""

    rule_id: str
    rule_version: str
    ruleset_version: str
    field: str
    result: str  # one of PASS / POTENTIAL_ISSUE / NOT_ASSESSABLE, enforced in __post_init__
    reason: str
    uncertainty_status: UncertaintyStatus
    evidence: List[Evidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.result not in _VALID_RESULTS:
            raise ValueError(
                f"Invalid rule result {self.result!r} for {self.rule_id}; "
                f"must be one of {sorted(_VALID_RESULTS)} — this engine never "
                f"produces a legal verdict."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleEvaluationResult:
    """The full set of findings for one inspection of one image."""

    source_image: Optional[str]
    ruleset_version: str
    context_used: InspectionContext
    findings: List[RuleFinding]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_image": self.source_image,
            "ruleset_version": self.ruleset_version,
            "context_used": self.context_used.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _ocr_reason(pipeline_result: PipelineResult) -> str:
    """A precise reason string distinguishing 'blocked by image quality'
    from 'OCR raised an exception' — never collapsed into one message, per
    the requirement that these stay distinguishable."""
    ocr = pipeline_result.ocr
    if ocr.reason:
        return f"OCR did not run: {ocr.reason}"
    if ocr.error:
        return f"OCR failed: {ocr.error}"
    return "OCR did not run and no reason was recorded"  # defensive fallback; should not occur


def _make_not_assessable(rule_id: str, field_name: str, reason: str, applicability_known: bool = True) -> RuleFinding:
    """The shared NOT_ASSESSABLE finding every rule returns when
    ``pipeline_result.ocr.ran`` is False — ``reason`` should come from
    ``_ocr_reason()`` so the "blocked by quality" vs "OCR raised an
    exception" distinction is preserved in the finding's own text."""
    return RuleFinding(
        rule_id=rule_id,
        rule_version=_RULE_VERSIONS[rule_id],
        ruleset_version=RULESET_VERSION,
        field=field_name,
        result=NOT_ASSESSABLE,
        reason=reason,
        uncertainty_status=UncertaintyStatus(
            ocr_ran=False, candidate_uncertain=None, applicability_known=applicability_known
        ),
        evidence=[],
    )


def _candidate_evidence(candidate: Dict[str, Any], value_key: Optional[str] = None) -> Evidence:
    """Build one Evidence entry from a raw extraction-candidate dict,
    copying its fields verbatim. ``value_key`` selects which candidate
    field is the "extracted value" for this rule's purposes (e.g.
    'parsed_value' for MRP); if absent/None, the candidate's own
    ``raw_text`` doubles as the extracted value."""
    extracted_value = candidate.get(value_key) if value_key else None
    if extracted_value is None and value_key is None:
        extracted_value = candidate.get("raw_text")
    return Evidence(
        source_image=candidate.get("source_image"),
        raw_text=candidate.get("raw_text"),
        extracted_value=extracted_value,
        confidence=candidate.get("confidence"),
        box=candidate.get("box"),
    )


def _is_food_category(context: InspectionContext) -> Optional[bool]:
    """True/False when the category is known and clearly food/non-food,
    None when unknown. Deliberately narrow (spec explicitly declines to
    invent a full category taxonomy): only an exact, case-insensitive
    "food" match counts as known-food; any other non-null category counts
    as known-non-food; a null category is unknown."""
    if context.category is None:
        return None
    return context.category.strip().lower() == "food"


# --------------------------------------------------------------------------
# LM-ENT-01 / LM-ENT-02 — Manufacturer / Packer / Importer (spec §7.2)
# --------------------------------------------------------------------------
def _evaluate_lm_ent_01(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-ENT-01"
    field_name = "manufacturer_packer_importer"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "entity_text") for c in candidates]

    if not candidates:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason="No manufacturer/packer/importer declaration found by OCR.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    any_confident = any(not c.get("uncertain", False) for c in candidates)
    if any_confident:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=PASS,
            reason="An identity declaration (manufacturer/packer/importer) was found in a recognizable format.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
            evidence=evidence,
        )

    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=POTENTIAL_ISSUE,
        reason="A label was detected but no confident identity value could be associated with it.",
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
        evidence=evidence,
    )


def _evaluate_lm_ent_02(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-ENT-02"
    field_name = "manufacturer_packer_importer"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "role") for c in candidates]

    if not candidates:
        # Nothing to assess a role for -- LM-ENT-01 already covers absence.
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=NOT_ASSESSABLE,
            reason="No entity declaration was found to assess a role for (see LM-ENT-01).",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    confirmed_roles = {"MANUFACTURER", "PACKER", "IMPORTER"}
    # Multiple confirmed roles (e.g. separate Manufacturer and Packer
    # declarations) is the NORMAL, fully-expected case under r.6(1)(a) --
    # not a conflict. Only flag when every candidate found is role=UNKNOWN
    # (e.g. "Marketed by" alone) -- that is never silently promoted to one
    # of the three confirmed roles.
    if any(c.get("role") in confirmed_roles for c in candidates):
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=PASS,
            reason="At least one entity declaration carries a confirmed role (Manufacturer/Packer/Importer).",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
            evidence=evidence,
        )

    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=POTENTIAL_ISSUE,
        reason=(
            "An entity declaration was found but its legal role (Manufacturer/Packer/Importer) "
            "could not be confirmed -- e.g. a 'Marketed by' declaration does not by itself satisfy "
            "this requirement and is never treated as one of the three confirmed roles."
        ),
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# LM-NAME-01 — Generic / Common Name (spec §7.3)
# --------------------------------------------------------------------------
def _evaluate_lm_name_01(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-NAME-01"
    field_name = "generic_common_name"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "product_name") for c in candidates]

    if not candidates:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason=(
                "No explicit generic/common name declaration found. Note: brand/marketing "
                "text is not treated as the generic name by design."
            ),
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    distinct_names = {c.get("product_name") for c in candidates if c.get("product_name")}
    if len(distinct_names) > 1:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason=(
                f"Multiple differing generic/common name declarations found ({sorted(distinct_names)}); "
                "cannot determine which applies -- not guessing."
            ),
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
            evidence=evidence,
        )

    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=PASS,
        reason="A generic/common name declaration was found in a recognizable format.",
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# LM-QTY-01 / LM-QTY-02 — Net Quantity (spec §7.4)
# --------------------------------------------------------------------------
def _evaluate_lm_qty_01(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-QTY-01"
    field_name = "net_quantity"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "quantities") for c in candidates]

    if not candidates:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason="No net-quantity declaration found by OCR.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    any_parsed = any(c.get("quantities") for c in candidates)
    if any_parsed:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=PASS,
            reason="A net-quantity declaration with a recognizable numeric value and unit was found.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
            evidence=evidence,
        )

    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=POTENTIAL_ISSUE,
        reason="A net-quantity label was found but no recognizable numeric value/unit could be parsed from it.",
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
        evidence=evidence,
    )


def _evaluate_lm_qty_02(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-QTY-02"
    field_name = "net_quantity"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "quantities") for c in candidates]

    # extraction.quantity already merges every representation of ONE
    # declaration (e.g. "1L/553g") into a single candidate's `quantities`
    # list -- a second, SEPARATE candidate in this list means the extractor
    # itself identified a genuinely distinct net-quantity declaration, not
    # just another representation of the same one.
    if len(candidates) > 1:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason=(
                f"{len(candidates)} separate net-quantity declarations were found; cannot determine "
                "which applies -- not guessing. (Multiple representations of ONE declaration, e.g. "
                "'1L/553g', are not a conflict and are preserved within a single candidate.)"
            ),
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
            evidence=evidence,
        )

    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=PASS,
        reason="No conflicting net-quantity declarations were found.",
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# LM-MRP-01 — MRP (spec §7.5; LM-MRP-02 intentionally not implemented)
# --------------------------------------------------------------------------
def _evaluate_lm_mrp_01(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-MRP-01"
    field_name = "mrp"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "parsed_value") for c in candidates]

    if not candidates:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason="No MRP declaration found by OCR.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    parsed_values = {c["parsed_value"] for c in candidates if c.get("parsed_value") is not None}

    if len(parsed_values) > 1:
        # Multiple candidates with genuinely differing numeric values: never
        # arbitrarily pick one as "the" MRP.
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason=(
                f"Multiple differing MRP values were found ({sorted(parsed_values)}); cannot determine "
                "which applies -- not guessing."
            ),
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
            evidence=evidence,
        )

    if len(parsed_values) == 1:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=PASS,
            reason="An MRP declaration with a recognizable numeric value was found.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
            evidence=evidence,
        )

    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=POTENTIAL_ISSUE,
        reason="An MRP label was found but no usable numeric price could be parsed from it.",
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# LM-DATE-01 / 02 / 03 — Best Before / Use By / Expiry (spec §7.6)
# --------------------------------------------------------------------------
def _evaluate_lm_date_01(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-DATE-01"
    field_name = "best_before_use_by"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "parsed_date") for c in candidates]

    if candidates:
        # A declaration exists -- evaluated regardless of category; a
        # present declaration never needs an applicability judgment to be
        # read.
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=PASS,
            reason="A Best Before/Use By/Expiry declaration was found.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
            evidence=evidence,
        )

    is_food = _is_food_category(context)
    if is_food is True:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=POTENTIAL_ISSUE,
            reason="No Best Before/Use By/Expiry declaration found, and the inspection context indicates a food product.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    reason = (
        "No Best Before/Use By/Expiry declaration found. This declaration is primarily a "
        "food-labelling (FSSAI) requirement, not a universal Legal Metrology one -- "
        + (
            "applicability could not be determined because no product category was supplied."
            if is_food is None
            else "the supplied category does not indicate a food product, so applicability of this "
            "declaration to this commodity is not established under this ruleset's current scope."
        )
    )
    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=NOT_ASSESSABLE,
        reason=reason,
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=is_food is not None),
        evidence=evidence,
    )


def _evaluate_lm_date_03(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-DATE-03"
    field_name = "best_before_use_by"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "parsed_date") for c in candidates]

    if not candidates:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=NOT_ASSESSABLE,
            reason="No date-type declaration was found to assess parseability for (see LM-DATE-01).",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    unparseable = [c for c in candidates if c.get("parsed_date") is None or c.get("uncertain")]
    if not unparseable:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=PASS,
            reason="All found date-type declaration(s) were confidently parsed.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
            evidence=evidence,
        )

    if len(unparseable) == len(candidates):
        reason = "A date-type label was found but the date itself could not be confidently read."
    else:
        reason = (
            "Some date-type declaration(s) could not be confidently read while others were "
            "-- both are preserved as evidence."
        )
    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=POTENTIAL_ISSUE,
        reason=reason,
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=True, applicability_known=True),
        evidence=evidence,
    )


def _evaluate_lm_date_02(pipeline_result: PipelineResult, context: InspectionContext) -> RuleFinding:
    rule_id = "LM-DATE-02"
    field_name = "best_before_use_by"
    if not pipeline_result.ocr.ran:
        return _make_not_assessable(rule_id, field_name, _ocr_reason(pipeline_result))

    candidates = (pipeline_result.declarations or {}).get(field_name, [])
    evidence = [_candidate_evidence(c, "parsed_date") for c in candidates]

    parseable = [c for c in candidates if c.get("parsed_date") is not None and not c.get("uncertain")]
    if not parseable:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=NOT_ASSESSABLE,
            reason="No confidently-parsed date is available to compare against the inspection date (see LM-DATE-01/03).",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=None, applicability_known=True),
            evidence=evidence,
        )

    if context.inspection_date is None:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=NOT_ASSESSABLE,
            reason=(
                "No inspection date was supplied in InspectionContext; an elapsed-date comparison "
                "requires an explicit date and this engine never invents one."
            ),
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=False),
            evidence=evidence,
        )

    elapsed = [c for c in parseable if date_type.fromisoformat(c["parsed_date"]) < context.inspection_date]
    if not elapsed:
        return RuleFinding(
            rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
            field=field_name, result=PASS,
            reason="All confidently-parsed date(s) are on or after the inspection date.",
            uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
            evidence=evidence,
        )

    # BEST_BEFORE and USE_BY/EXPIRY carry different practical significance
    # (quality vs. safety) and are never collapsed into one message.
    elapsed_field_names = {c.get("field_name") for c in elapsed}
    messages = []
    if elapsed_field_names & {"EXPIRY", "USE_BY"}:
        messages.append(
            "Potential issue -- declared use-by/expiry date appears to have elapsed. "
            "Inspector verification required."
        )
    if "BEST_BEFORE" in elapsed_field_names:
        messages.append(
            "Potential issue -- declared best-before date appears to have elapsed. Note: "
            "'best before' relates to peak quality, not necessarily food safety; inspector "
            "verification required."
        )
    return RuleFinding(
        rule_id=rule_id, rule_version=_RULE_VERSIONS[rule_id], ruleset_version=RULESET_VERSION,
        field=field_name, result=POTENTIAL_ISSUE,
        reason=" ".join(messages),
        uncertainty_status=UncertaintyStatus(ocr_ran=True, candidate_uncertain=False, applicability_known=True),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
_RULE_ORDER = (
    "LM-ENT-01", "LM-ENT-02", "LM-NAME-01",
    "LM-QTY-01", "LM-QTY-02", "LM-MRP-01",
    "LM-DATE-01", "LM-DATE-02", "LM-DATE-03",
)

_EVALUATORS = {
    "LM-ENT-01": _evaluate_lm_ent_01,
    "LM-ENT-02": _evaluate_lm_ent_02,
    "LM-NAME-01": _evaluate_lm_name_01,
    "LM-QTY-01": _evaluate_lm_qty_01,
    "LM-QTY-02": _evaluate_lm_qty_02,
    "LM-MRP-01": _evaluate_lm_mrp_01,
    "LM-DATE-01": _evaluate_lm_date_01,
    "LM-DATE-02": _evaluate_lm_date_02,
    "LM-DATE-03": _evaluate_lm_date_03,
}


def evaluate(pipeline_result: PipelineResult, context: Optional[InspectionContext] = None) -> RuleEvaluationResult:
    """Evaluate all nine v1 rules against one ``PipelineResult``.

    Deterministic: the same ``pipeline_result`` + the same ``context``
    always produces the same findings (no wall-clock reads, no randomness,
    no network). Exactly one :class:`RuleFinding` is produced per rule,
    in the fixed order above, regardless of the pipeline's outcome (hard
    image-quality block, OCR failure, zero detections, or full success) --
    every finding's ``result``/``reason``/``uncertainty_status`` says which
    of those happened.

    This function does not run OCR, does not call
    ``pipeline.run_inspection_pipeline``, and does not mutate
    ``pipeline_result`` in any way.
    """
    ctx = context if context is not None else InspectionContext()
    findings = [_EVALUATORS[rule_id](pipeline_result, ctx) for rule_id in _RULE_ORDER]
    return RuleEvaluationResult(
        source_image=pipeline_result.source_image,
        ruleset_version=RULESET_VERSION,
        context_used=ctx,
        findings=findings,
    )
