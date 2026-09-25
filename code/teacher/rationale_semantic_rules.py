"""State-conditioned semantic checks for short ThyQC rationale atoms.

The checks are intentionally task-local.  They ensure that each rationale
supports its stored teacher state without turning the five-task decision chain
into a hard one-to-one map.
"""
from __future__ import annotations

import re


TASKS = (
    "image_usability",
    "structure_visibility",
    "keyframe_status",
    "primary_defect",
    "recommended_action",
)


def _text(value):
    return re.sub(r"\s+", " ", str(value or "").lower().replace("_", " ")).strip()


def _has(text, *patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def _error(task, state, reason):
    return f"{task}:{state}:{reason}"


def semantic_alignment_errors(rationale, states):
    """Return task-local label/rationale contradictions.

    This does not infer one task from another.  In particular, a partial view
    may coexist with primary_defect=none and a rescan recommendation, provided
    each task's own rationale explains its own state.
    """
    errors = []
    values = {task: _text((rationale or {}).get(task)) for task in TASKS}
    state = {task: _text((states or {}).get(task)) for task in TASKS}

    text = values["image_usability"]
    label = state["image_usability"]
    if label == "acceptable":
        if not _has(text, r"\bretain", r"\breadab", r"\binterpret", r"\bdiscern",
                    r"\bpreserv", r"\bstable signal", r"\bcoherent texture", r"\brecognizable",
                    r"\bresolv", r"\bsharp", r"\bclear"):
            errors.append(_error("image_usability", label, "missing_positive_usability_evidence"))
        if _has(text, r"\bprevent(?:s|ed)? reliable", r"\boverwhelm", r"\bsignal failure dominates",
                r"\bbroad (?:visual |field )?corruption", r"\buninterpretable"):
            errors.append(_error("image_usability", label, "claims_unusable_image"))
    elif label == "limited":
        if not _has(text, r"\bbut\b", r"\balthough\b", r"\bwhile\b", r"\breduc",
                    r"\bweak", r"\bvar(?:y|ies|iation)", r"\buneven", r"\blower",
                    r"\badequately .* but", r"\badequately bright"):
            errors.append(_error("image_usability", label, "missing_mixed_quality_evidence"))
        if _has(text, r"\bprevent(?:s|ed)? reliable (?:anatomical )?interpret", r"\boverwhelms? anatomy"):
            errors.append(_error("image_usability", label, "claims_complete_unusability"))
    elif label == "unacceptable":
        if not _has(text, r"\bcorrupt", r"\bdropout", r"\bdistort", r"\bobscur",
                    r"\bprevent", r"\bunreliable", r"\boverwhelm", r"\bsignal failure", r"\bhides? tissue",
                    r"\bexcessively dark", r"\blacks? (?:usable )?(?:tissue )?contrast"):
            errors.append(_error("image_usability", label, "missing_failure_evidence"))

    text = values["structure_visibility"]
    label = state["structure_visibility"]
    target = _has(text, r"\bthyroid", r"\bgland", r"\bparenchyma", r"\btarget",
                  r"\bmargin", r"\bboundar", r"\btissue")
    if not target:
        errors.append(_error("structure_visibility", label, "missing_target_anatomy"))
    if label == "adequate":
        if not _has(text, r"\btraceab", r"\bidentif", r"\bvisible", r"\bcomplete",
                    r"\bcontinuously represented", r"\bpreserv"):
            errors.append(_error("structure_visibility", label, "missing_complete_visibility_evidence"))
        if _has(text, r"\bcut off", r"\bomits?", r"\bmissing", r"\bno (?:reliable|identifiable)", r"\bunreliable"):
            errors.append(_error("structure_visibility", label, "claims_missing_structure"))
    elif label == "partial":
        if not _has(text, r"\bonly part", r"\bone .*boundar", r"\bcut off", r"\bmissing",
                    r"\bomits?", r"\bincomplete", r"\blacks? (?:full|complete)", r"\bpartly",
                    r"\bvisible .* while .*margin", r"\bcoverage misses? a boundary",
                    r"\bboundar(?:y|ies) exits? the field", r"\boutside the field",
                    r"\bvisible .* while .*coverage var(?:y|ies)"):
            errors.append(_error("structure_visibility", label, "missing_partial_coverage_evidence"))
        if _has(text, r"\bno (?:reliable|identifiable|recognizable) (?:thyroid|gland|parenchyma|tissue)"):
            errors.append(_error("structure_visibility", label, "claims_absent_structure"))
    elif label == "absent":
        if not _has(text, r"\bno (?:reliable|identifiable|recognizable|confirmable)",
                    r"\bno (?:thyroid |gland )?(?:parenchyma|tissue|target) is identifiable",
                    r"\bno detector-supported (?:thyroid |gland )?(?:parenchyma|tissue|target) is present",
                    r"\bno detector-supported .*can be (?:identified|confirmed)",
                    r"\blacks? recognizable", r"\bcannot be (?:identified|confirmed|separated)",
                    r"\bnot identifiable", r"\bmissing thyroid", r"\bno interpretable thyroid"):
            errors.append(_error("structure_visibility", label, "missing_absence_evidence"))
        if _has(text, r"\bremains? (?:visible|traceable)", r"\bcomplete (?:field )?coverage"):
            errors.append(_error("structure_visibility", label, "claims_visible_structure"))
    elif label == "uncertain":
        if not _has(text, r"\bunreliable", r"\binconsistent", r"\bobscur", r"\bprevent(?:s|ed)? confident",
                    r"\bweak .*support", r"\bvariable", r"\bcannot .*confident",
                    r"\bcannot be followed", r"\bcannot .*consistently",
                    r"\bnot consistently (?:identifiable|visible|traceable)"):
            errors.append(_error("structure_visibility", label, "missing_uncertainty_evidence"))

    text = values["keyframe_status"]
    label = state["keyframe_status"]
    if not _has(text, r"\bframe", r"\bsequence", r"\bview", r"\bcross-frame"):
        errors.append(_error("keyframe_status", label, "missing_frame_evidence"))
    if label == "yes":
        if not _has(text, r"\bstable", r"\brepresent", r"\bcomplete", r"\bfull", r"\bentire"):
            errors.append(_error("keyframe_status", label, "missing_positive_keyframe_evidence"))
        if _has(text, r"\bno frame", r"\blacks? (?:full|complete)", r"\bomits? part", r"\binconsistent"):
            errors.append(_error("keyframe_status", label, "claims_nonrepresentative_frame"))
    elif label == "candidate":
        if not _has(text, r"\bbest", r"\bpotential", r"\brepresent", r"\bstability",
                    r"\bcoverage changes", r"\bvaries across"):
            errors.append(_error("keyframe_status", label, "missing_candidate_frame_evidence"))
        if not _has(text, r"\bbut\b", r"\blacks?", r"\bomits?", r"\bvar(?:y|ies|iation)",
                    r"\binconsistent", r"\bincomplete", r"\bstill", r"\bchanges"):
            errors.append(_error("keyframe_status", label, "missing_candidate_limitation"))
    elif label == "no":
        if not _has(text, r"\bno (?:sampled )?frame", r"\blacks? a frame", r"\bprevents? selection",
                    r"\bno view", r"\bcross-frame corruption"):
            errors.append(_error("keyframe_status", label, "missing_negative_keyframe_evidence"))

    text = values["primary_defect"]
    label = state["primary_defect"]
    if label == "none":
        if not _has(text, r"\bno (?:single |dominant |major )", r"\bfree of a dominant",
                    r"\bwithout (?:a )?dominant", r"\bdoes not contain a dominant",
                    r"\bno .*\b(?:defect|problem|failure|corruption|coverage loss)\b"):
            errors.append(_error("primary_defect", label, "missing_explicit_no_dominant_problem"))
        if _has(text, r"^(?!.*\bno\b).*\bdominates?\b", r"\bmain (?:defect|limitation|problem)",
                r"\bmissing .* is the dominant", r"\bprevents? complete", r"\bcuts? off the thyroid"):
            errors.append(_error("primary_defect", label, "claims_dominant_problem"))
    elif label == "severe artifact":
        if not _has(text, r"\bcorrupt", r"\bdropout", r"\bdistort", r"\bsignal failure",
                    r"\bstreak", r"\bobscur"):
            errors.append(_error("primary_defect", label, "missing_corruption_evidence"))
        if not _has(text, r"\bdomin", r"\bmain", r"\bfield-wide", r"\bbroad", r"\bsevere", r"\bprevents?"):
            errors.append(_error("primary_defect", label, "missing_dominance_evidence"))
    elif label == "thyroid not visible":
        if not _has(text, r"\bno\b", r"\bnot\b", r"\bmissing", r"\black") or not _has(
                text, r"\bthyroid", r"\bgland", r"\bparenchyma", r"\blocaliz", r"\btarget"):
            errors.append(_error("primary_defect", label, "missing_target_localization_failure"))
    elif label == "target incomplete":
        if not _has(text, r"\bmissing", r"\bcut off", r"\btruncat", r"\bincomplete", r"\bincomplet",
                    r"\bomits?", r"\blacks?", r"\boutside the field"):
            errors.append(_error("primary_defect", label, "missing_incomplete_target_evidence"))
        if not _has(text, r"\bboundar", r"\bmargin", r"\bcoverage", r"\bextent", r"\bfield"):
            errors.append(_error("primary_defect", label, "missing_coverage_object"))

    text = values["recommended_action"]
    label = state["recommended_action"]
    if label == "hold":
        if not _has(text, r"\bretain", r"\bpreserv", r"\bsufficient", r"\bno additional",
                    r"\bcan be retained", r"\balready .* (?:coverage|context|evidence)"):
            errors.append(_error("recommended_action", label, "missing_retention_objective"))
        requests_more = _has(
            text, r"\bnew (?:imaging|acquisition|frame|angle)", r"\bfresh", r"\breimag",
            r"(?<!no )\badditional .*needed",
        )
        if requests_more and not _has(text, r"\bno additional", r"\bwithout additional"):
            errors.append(_error("recommended_action", label, "requests_new_acquisition"))
    elif label == "repeat acquisition":
        if not _has(text, r"\bnew imaging", r"\bnew acquisition", r"\bfresh imaging", r"\breimag",
                    r"\breplace", r"\brestore", r"\brecover"):
            errors.append(_error("recommended_action", label, "missing_repeat_objective"))
    elif label == "reacquire target":
        if not _has(text, r"\bnew", r"\bfresh", r"\breacquir", r"\bacquir", r"\btarget", r"\blocaliz",
                    r"\brestore", r"\bcapture") or not _has(
                text, r"\bthyroid", r"\btissue", r"\btarget", r"\bgland", r"\bfield"):
            errors.append(_error("recommended_action", label, "missing_target_recovery_objective"))
    elif label == "adjust angle and rescan":
        if not _has(text, r"\bangle", r"\bangled", r"\bcoverage", r"\bcapture", r"\binclude",
                    r"\brestore", r"\brecover", r"\bfield"):
            errors.append(_error("recommended_action", label, "missing_coverage_adjustment_objective"))

    return errors


def state_condition_guidance(states):
    """Compact generation constraints derived from each stored state."""
    state = {task: _text((states or {}).get(task)) for task in TASKS}
    guidance = []
    if state["primary_defect"] == "none":
        guidance.append("Slot 4 must explicitly say that no single dominant acquisition problem is present")
    elif state["primary_defect"] == "severe artifact":
        guidance.append("Slot 4 must identify field-wide signal corruption as the dominant problem")
    elif state["primary_defect"] == "thyroid not visible":
        guidance.append("Slot 4 must explain failed thyroid localization")
    elif state["primary_defect"] == "target incomplete":
        guidance.append("Slot 4 must identify a missing or truncated thyroid boundary")
    if state["structure_visibility"] == "partial":
        guidance.append(
            "Slot 2 must name visible tissue and the missing or truncated thyroid boundary or extent"
        )
    if state["keyframe_status"] == "candidate":
        guidance.append("Slot 3 must name both the best frame evidence and its remaining limitation")
    if state["recommended_action"] == "hold":
        guidance.append("Slot 5 must retain current coverage and must not request new imaging")
    return "; ".join(guidance) + ("." if guidance else "")
