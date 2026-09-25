"""Generate short, image-grounded task rationales with the trained LLaVA-Med teacher."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from peft import PeftModel
from rationale_semantic_rules import semantic_alignment_errors, state_condition_guidance
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)


TASKS = (
    "image_usability",
    "structure_visibility",
    "keyframe_status",
    "primary_defect",
    "recommended_action",
)

VISUAL_TASKS = TASKS[:4]

# Exact origin of the self-built ThyQC RAG used by the historical teacher
# ablation.  It was a seven-rule clinical-QC prompt knowledge base, not a
# generic external vector store.  Only its hard defect->action mapping is
# softened here, as required by the rationale-aware G2D formulation.
RAG_PROVENANCE = {
    "source_file": "code/teacher_ft/run_llava_med_ablation_v1_21.py",
    "source_variant": "A3_prompt_checklist_rag",
    "original_rule_count": 7,
    "adaptation": "soft_action_only",
    "retrieval_mode": "sample_specific_state_and_low_level_route",
    "historical_evidence_override": "paper_eval_100 evidence_basis when available",
}

# Recovered from the original A3 rules and the later sample-level evidence
# audit.  The historical identifiers and clinical-QC roles are preserved.
RAG_KNOWLEDGE = (
    {
        "id": "RAG-C-STRUCTURE-USABLE",
        "tasks": ("structure_visibility", "keyframe_status"),
        "text": (
            "Usable thyroid coverage requires recognizable parenchyma, traceable "
            "target boundaries, and sufficient surrounding context across frames."
        ),
        "source_ids": ("teacher_v1_11_efficient_composite", "visual_evidence_final"),
    },
    {
        "id": "RAG-E-HOLD-USABLE",
        "tasks": ("recommended_action",),
        "text": (
            "When image quality, target coverage, and frame stability are jointly "
            "sufficient, retaining the current acquisition may be appropriate."
        ),
        "source_ids": ("teacher_v1_11_efficient_composite",),
    },
    {
        "id": "RAG-B-SEVERE-ARTIFACT",
        "tasks": ("image_usability", "primary_defect", "recommended_action"),
        "text": (
            "Severe corruption, bright-screen failure, or signal loss can obscure "
            "anatomy; the acquisition response should reflect the remaining evidence."
        ),
        "source_ids": ("teacher_v1_11_efficient_composite", "visual_evidence_final"),
    },
    {
        "id": "RAG-C-TARGET-COMPLETE",
        "tasks": ("structure_visibility", "keyframe_status"),
        "text": (
            "Target completeness is supported when the relevant thyroid extent and "
            "boundaries remain represented across the sampled frames."
        ),
        "source_ids": ("two_stage_images_incomplete_recovery", "visual_evidence_final"),
    },
    {
        "id": "RAG-D-TARGET-INCOMPLETE",
        "tasks": ("primary_defect", "recommended_action"),
        "text": (
            "A missing or cut-off thyroid boundary indicates incomplete coverage; "
            "additional viewing angle or field coverage may resolve the omission."
        ),
        "source_ids": ("two_stage_images_incomplete_recovery",),
    },
    {
        "id": "RAG-C-THYROID-VISIBILITY",
        "tasks": ("structure_visibility", "primary_defect", "recommended_action"),
        "text": (
            "Thyroid visibility requires identifiable parenchyma rather than screen "
            "layout or boxes; weak visual support may reflect region or quality limits."
        ),
        "source_ids": ("two_stage_notvisible_recall_boundary", "yolo_presence_features"),
    },
    {
        "id": "RAG-E-DEFECT-ACTION",
        "tasks": ("recommended_action",),
        "text": (
            "Choose the action from joint evidence about usability, visibility, "
            "keyframe stability, defect type, and uncertainty; no defect implies a "
            "single mandatory action."
        ),
        "source_ids": ("teacher_v1_11_efficient_composite",),
    },
)

EVIDENCE_KEYS = {
    "image_usability": "usability_evidence",
    "structure_visibility": "visibility_evidence",
    "keyframe_status": "keyframe_evidence",
    "primary_defect": "defect_evidence",
    "recommended_action": "action_evidence",
}

EVIDENCE_ALIASES = {
    "image_usability": ("usability_evidence", "image_usability_evidence", "image_usability"),
    "structure_visibility": ("visibility_evidence", "structure_visibility_evidence", "structure_visibility"),
    "keyframe_status": ("keyframe_evidence", "keyframe_status_evidence", "keyframe_status"),
    "primary_defect": ("defect_evidence", "primary_defect_evidence", "primary_defect"),
    "recommended_action": ("action_evidence", "recommended_action_evidence", "recommended_action"),
}

OBSERVATION_TEXT = {
    "severe_artifact_pattern": "broad field corruption pattern",
    "no_signal_or_bright_screen": "signal dropout or bright-screen pattern",
    "anatomy_not_assessable": "anatomical tissue cannot be reliably assessed",
    "thyroid_region_visible": "recognizable thyroid-region tissue",
    "target_complete": "target boundaries remain represented",
    "usable_keyframe": "stable representative frame coverage",
    "thyroid_region_partly_visible": "only part of the thyroid region remains visible",
    "incomplete_target_coverage": "one or more target boundaries are cut off",
    "no_reliable_thyroid_parenchyma": "no reliable thyroid parenchyma is identifiable",
    "weak_or_absent_yolo_support": "weak or missing detector support for the thyroid region",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm(value):
    return str(value or "").strip()


def render_observation_hint(value):
    return OBSERVATION_TEXT.get(value, value.replace("_", " "))


def rationale_words(value):
    """Count hyphenated clinical/technical compounds as single words."""
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", norm(value))


def teacher_index(path):
    index = {}
    for row in read_jsonl(path):
        for key in ("sample_id", "segment_id", "review_id"):
            value = norm(row.get(key))
            if value:
                index[value] = row
    return index


def rag_evidence_index(path):
    """Index the original ThyQC sample-level RAG evidence records."""
    index = {}
    if not path:
        return index
    for row in read_jsonl(path):
        payload = row.get("teacher_label_full", {}).get("evidence_basis", {})
        if not payload:
            payload = row.get("evidence_basis", {})
        if not payload:
            continue
        for key in ("review_id", "sample_id", "segment_id"):
            value = norm(row.get(key))
            if value:
                index[value] = payload
    return index


def assess_teacher_knowledge_consistency(states, visual_observations):
    """Return whether a teacher package is safe for semantic transport.

    This is deliberately a narrow *conflict* gate, not another hard clinical
    decision map.  Alternative actions remain allowed.  We only reject direct
    contradictions inside the teacher decisions or between those decisions
    and the retrieved legacy ThyQC evidence.
    """
    state = {task: norm(states.get(task)).lower() for task in TASKS}
    observations = set(visual_observations or ())
    reasons = []

    if state["structure_visibility"] == "absent":
        if state["primary_defect"] == "none":
            reasons.append("absent_structure_with_no_defect")
        if state["keyframe_status"] in {"yes", "candidate"}:
            reasons.append("absent_structure_with_positive_keyframe")
    if state["image_usability"] == "unacceptable":
        if state["primary_defect"] == "none":
            reasons.append("unacceptable_image_with_no_defect")

    action = state["recommended_action"]
    if (
        state["image_usability"] == "acceptable"
        and state["structure_visibility"] == "adequate"
        and state["keyframe_status"] == "yes"
        and state["primary_defect"] == "none"
        and action != "hold"
    ):
        reasons.append("fully_sufficient_evidence_with_new_acquisition")
    if (
        state["structure_visibility"] == "absent"
        and state["keyframe_status"] == "no"
        and state["primary_defect"] == "thyroid_not_visible"
        and action == "hold"
    ):
        reasons.append("absent_target_with_hold_action")

    # RAG observations and Recommended Action are deliberately excluded from
    # this hard gate.  RAG supplies clinical semantics and Action is a soft
    # decision conditioned on the first four tasks; neither may overwrite the
    # teacher's stored state or impose a one-to-one defect/action map.

    return not reasons, reasons


def unique_requests(manifest, teacher_jsonl, rag_evidence_jsonl=""):
    teachers = teacher_index(teacher_jsonl)
    historical_rag = rag_evidence_index(rag_evidence_jsonl)
    requests = {}
    for row in read_jsonl(manifest):
        aliases = [
            norm(row.get("original_sample_id")),
            norm(row.get("segment_id")),
            norm(row.get("review_id")),
            norm(row.get("sample_id")),
        ]
        canonical = next((x for x in aliases[:2] if x), aliases[-1])
        if canonical in requests:
            continue
        teacher = next((teachers[x] for x in aliases if x in teachers), None)
        if teacher is not None:
            states = teacher.get("raw_prediction") or teacher.get("teacher_label")
            probabilities = teacher.get("probabilities") or {}
            auxiliary_context = teacher.get("auxiliary_context", "")
        else:
            states = row["student_label"]
            probabilities = {}
            auxiliary_context = row.get("auxiliary_context", "")
        paths = row.get("input_paths") or [row.get("input_path")]
        image_path = paths[0]
        low_level = low_level_information(image_path, auxiliary_context)
        evidence_basis = next(
            (historical_rag[x] for x in aliases if x in historical_rag), {}
        )
        historical_rule_ids = set(evidence_basis.get("retrieved_rule_ids") or [])
        rag_entries = retrieve_rag_entries(states, low_level, historical_rule_ids)
        visual_observations = list(evidence_basis.get("visual_observation") or [])
        if not visual_observations:
            visual_observations = derive_rag_visual_observations(states, low_level)
        knowledge_usable, conflict_reasons = assess_teacher_knowledge_consistency(
            states, visual_observations
        )
        requests[canonical] = {
            "sample_id": canonical,
            "segment_id": norm(row.get("segment_id")) or canonical,
            "review_id": norm(row.get("review_id")),
            "image_path": image_path,
            "states": {task: states[task] for task in TASKS},
            "probabilities": {task: probabilities.get(task, {}) for task in TASKS},
            "low_level_information": low_level,
            "low_level_summary": summarize_low_level_information(low_level),
            "rag_entries": rag_entries,
            "rag_visual_observations": visual_observations,
            "rag_computed_metrics": list(evidence_basis.get("computed_metrics") or []),
            "historical_rag_source_ids": list(evidence_basis.get("source_ids") or []),
            "state_source": "teacher_raw_prediction" if teacher is not None else "physician_label_for_uncovered_sample",
            "teacher_knowledge_usable": knowledge_usable,
            "teacher_knowledge_conflicts": conflict_reasons,
        }
    return list(requests.values())


def patch_processor(processor, config):
    if getattr(processor, "patch_size", None) is None and hasattr(config, "vision_config"):
        processor.patch_size = getattr(config.vision_config, "patch_size", None)
    processor.vision_feature_select_strategy = getattr(
        config, "vision_feature_select_strategy", "default"
    )
    processor.num_additional_image_tokens = 1
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def entropy_u8(arr):
    hist = np.bincount(arr.reshape(-1), minlength=256).astype(np.float64)
    probabilities = hist / max(1.0, hist.sum())
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum() / 8.0)


def parse_auxiliary_context(auxiliary_context):
    if isinstance(auxiliary_context, dict):
        return dict(auxiliary_context)
    text = norm(auxiliary_context)
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def low_level_information(image_path, auxiliary_context=""):
    """Objective image descriptors; no label or clinical decision is inferred."""
    image = Image.open(image_path).convert("L").resize((256, 256))
    array = np.asarray(image, dtype=np.float32)
    array_u8 = array.astype(np.uint8)
    dx = np.diff(array, axis=1)
    dy = np.diff(array, axis=0)
    gradient = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)
    edge = np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    valid = (array > 8) & (array < 248)
    metrics = {
        "brightness_mean": float(array.mean() / 255.0),
        "contrast_std": float(array.std() / 255.0),
        "valid_intensity_ratio": float(valid.mean()),
        "dark_ratio": float((array < 15).mean()),
        "bright_ratio": float((array > 240).mean()),
        "entropy_normalized": entropy_u8(array_u8),
        "gradient_mean": float(gradient.mean() / 255.0),
        "edge_mean": float(edge.mean() / 255.0),
    }
    auxiliary = parse_auxiliary_context(auxiliary_context)
    for key in ("det_count", "max_conf", "mean_conf", "mean_brightness_valid_ratio"):
        if key in auxiliary:
            metrics[key] = auxiliary[key]
    return metrics


def summarize_low_level_information(metrics):
    """Turn objective metrics into evidence phrases, never class decisions."""
    summary = []
    brightness = float(metrics.get("brightness_mean", 0.5))
    contrast = float(metrics.get("contrast_std", 0.2))
    gradient = float(metrics.get("gradient_mean", 0.08))
    dark_ratio = float(metrics.get("dark_ratio", 0.0))
    bright_ratio = float(metrics.get("bright_ratio", 0.0))
    det_count = metrics.get("det_count")
    max_conf = metrics.get("max_conf")

    if brightness < 0.28:
        summary.append("overall field is relatively dark")
    elif brightness > 0.70:
        summary.append("overall field is relatively bright")
    else:
        summary.append("overall brightness lies in the mid range")
    if contrast < 0.16:
        summary.append("grayscale contrast is narrow")
    elif contrast > 0.30:
        summary.append("strong intensity variation crosses the field")
    else:
        summary.append("grayscale contrast is moderately distributed")
    if gradient < 0.035:
        summary.append("fine boundary gradients are weak")
    elif gradient > 0.11:
        summary.append("local edge variation is pronounced")
    else:
        summary.append("boundary gradients remain moderately expressed")
    if dark_ratio > 0.45:
        summary.append("a large dark region occupies the contact sheet")
    if bright_ratio > 0.35:
        summary.append("a large bright region occupies the contact sheet")
    if det_count is not None:
        det_count = int(det_count)
        if det_count == 0:
            summary.append("the thyroid-region detector provides no supporting box")
        else:
            confidence = float(max_conf or 0.0)
            strength = "strong" if confidence >= 0.70 else "weak"
            summary.append(
                f"{det_count} sampled frames have {strength} detector support"
            )
    return summary


def derive_rag_visual_observations(states, low_level):
    """Restore the original ThyQC RAG observation vocabulary when history is absent."""
    defect = norm(states.get("primary_defect")).lower()
    if defect == "severe_artifact":
        observations = ["severe_artifact_pattern", "anatomy_not_assessable"]
        if (
            float(low_level.get("bright_ratio", 0.0)) > 0.20
            or float(low_level.get("dark_ratio", 0.0)) > 0.35
        ):
            observations.insert(1, "no_signal_or_bright_screen")
        return observations
    visibility = norm(states.get("structure_visibility")).lower()
    if defect == "target_incomplete" or visibility == "partial":
        return ["thyroid_region_partly_visible", "incomplete_target_coverage"]
    if defect == "thyroid_not_visible" or visibility == "absent":
        return ["no_reliable_thyroid_parenchyma", "weak_or_absent_yolo_support"]
    return ["thyroid_region_visible", "target_complete", "usable_keyframe"]


def task_observation_hints(task, observations):
    """Route original RAG observations to the task they can actually support."""
    observation_map = {
        "image_usability": {
            "severe_artifact_pattern", "no_signal_or_bright_screen",
            "thyroid_region_visible", "thyroid_region_partly_visible",
        },
        "structure_visibility": {
            "anatomy_not_assessable", "thyroid_region_visible",
            "thyroid_region_partly_visible", "incomplete_target_coverage",
            "no_reliable_thyroid_parenchyma", "weak_or_absent_yolo_support",
            "target_complete",
        },
        "keyframe_status": {
            "anatomy_not_assessable", "usable_keyframe", "target_complete",
            "incomplete_target_coverage", "no_reliable_thyroid_parenchyma",
        },
        "primary_defect": {
            "severe_artifact_pattern", "no_signal_or_bright_screen",
            "incomplete_target_coverage", "no_reliable_thyroid_parenchyma",
            "target_complete",
        },
        "recommended_action": set(observations),
    }
    allowed = observation_map[task]
    return [item for item in observations if item in allowed]


def task_low_level_hints(task, summary):
    if task == "image_usability":
        tokens = ("brightness", "contrast", "intensity", "gradient", "dark", "bright")
    elif task == "structure_visibility":
        tokens = ("detector", "boundary", "frame")
    elif task == "keyframe_status":
        tokens = ("frames", "detector", "boundary")
    elif task == "primary_defect":
        tokens = ("intensity", "gradient", "dark", "bright", "detector")
    else:
        return list(summary)
    return [item for item in summary if any(token in item for token in tokens)]


def retrieve_rag_entries(states, low_level, preferred_rule_ids=None):
    """Recover sample-specific legacy rule IDs without using a hard action map."""
    preferred = set(preferred_rule_ids or ())
    if preferred:
        ids = preferred | {"RAG-E-DEFECT-ACTION"}
        return [entry for entry in RAG_KNOWLEDGE if entry["id"] in ids]

    ids = {"RAG-E-DEFECT-ACTION"}
    usability = norm(states.get("image_usability")).lower()
    visibility = norm(states.get("structure_visibility")).lower()
    keyframe = norm(states.get("keyframe_status")).lower()
    defect = norm(states.get("primary_defect")).lower()
    # Preserve the original RAG's mutually exclusive primary evidence routes.
    if defect == "severe_artifact" or usability == "unacceptable":
        ids.add("RAG-B-SEVERE-ARTIFACT")
    elif defect == "target_incomplete":
        ids.update(("RAG-C-TARGET-COMPLETE", "RAG-D-TARGET-INCOMPLETE"))
    elif defect == "thyroid_not_visible":
        ids.add("RAG-C-THYROID-VISIBILITY")
    elif visibility == "adequate" and keyframe == "yes" and defect == "none":
        ids.update(("RAG-C-STRUCTURE-USABLE", "RAG-E-HOLD-USABLE"))
    elif visibility in {"absent", "uncertain"}:
        ids.add("RAG-C-THYROID-VISIBILITY")
    elif visibility == "partial":
        ids.update(("RAG-C-TARGET-COMPLETE", "RAG-D-TARGET-INCOMPLETE"))
    if not preferred and float(low_level.get("bright_ratio", 0.0)) > 0.80:
        ids.add("RAG-B-SEVERE-ARTIFACT")
    return [entry for entry in RAG_KNOWLEDGE if entry["id"] in ids]


def prompt(states, low_level_summary, rag_entries, rag_visual_observations=None):
    low_level_text = "\n".join(f"- {item}" for item in low_level_summary)
    observations = rag_visual_observations or []
    observation_text = "\n".join(
        f"- {render_observation_hint(item)}" for item in observations
    ) or "- no archived sample-specific observation; inspect the image directly"
    rag_text = "\n".join(f"- [{entry['id']}] {entry['text']}" for entry in rag_entries)
    decision_text = "\n".join(
        f"- {task.replace('_', ' ')}: {states[task]}" for task in TASKS
    )
    state_guidance = state_condition_guidance(states)
    return f"""You are the fine-tuned thyroid-ultrasound quality-control teacher.
Inspect the contact sheet and use the auxiliary context below.
The teacher decisions are control conditions identifying what each rationale must explain.
Do not copy these labels into the rationale text. Probabilities are stored separately
and are not provided here; they define transport mass, not semantic geometry.
Generate only the visual evidence package used for semantic geometry.
Do not infer diagnosis, pathology, TI-RADS, patient identity, or facts not visible in the image.
Internally reason in this order: image usability → structure visibility → keyframe status → primary defect → recommended action.
Use the first four evidence items when forming the action evidence, but do not force a one-to-one defect/action rule.
Do not output the intermediate reasoning or a long chain of thought.
For each of the five tasks, write one sample-specific rationale of 5-15 words.
Do not infer or state any classification label. Do not give follow-up, treatment, or diagnostic advice.
For the action item, explain the acquisition objective supported by the joint evidence.
Every string must name a concrete visible attribute from this contact sheet.
Do not use the words decision, status, class, label, acceptable, unacceptable,
limited, adequate, partial, absent, uncertain, or candidate.
Do not copy a generic sentence across tasks; inspect each requested aspect separately.
Keep the entire JSON response concise.

Keep the five output slots semantically separate:
- Slot 1 describes only clarity, brightness, contrast, signal, or corruption; never coverage.
- Slot 2 describes only thyroid tissue, margins, visibility, or field coverage.
- Slot 3 describes only frame stability and representative target coverage.
- Slot 4 describes only the dominant acquisition problem, without copying Slot 1 or Slot 2.
- Slot 5 describes only the acquisition objective supported by Slots 1-4.

Teacher decisions are control conditions only (explain, never copy):
{decision_text}
State-conditioned semantic requirements:
{state_guidance or '- Support every stored decision without inventing a stronger defect.'}

Objective low-level visual summary (evidence only, never labels):
{low_level_text}

Original ThyQC sample-level RAG observations (retrieval hints, verify visually):
{observation_text}

Retrieved legacy ThyQC clinical-QC knowledge:
{rag_text}

Return one JSON array only, with five strings in exactly this order:
[image usability evidence, structure visibility evidence, keyframe evidence,
 primary defect evidence, acquisition-objective evidence]"""


def state_aware_retry_guidance(states):
    """Clarify task duties when a first generation conflicts with its controls."""
    notes = []
    defect = norm(states.get("primary_defect")).lower()
    action = norm(states.get("recommended_action")).lower()
    if defect == "none":
        notes.append(
            "The primary-defect control is none: describe that no dominant "
            "corruption or coverage loss is present. Do not promote an isolated "
            "brightness or contrast descriptor into a primary defect."
        )
    if action == "hold":
        notes.append(
            "The action control is hold: explain why the current acquisition and "
            "target coverage can be retained; do not request new or additional imaging."
        )
    if defect == "severe_artifact":
        notes.append(
            "Keep the primary-defect evidence distinct from image-usability wording: "
            "name the dominant field-wide signal failure rather than copying Slot 1."
        )
    if defect == "thyroid_not_visible":
        notes.append(
            "Keep structure and defect distinct: Slot 2 describes absent thyroid "
            "evidence, while Slot 4 describes failed target localization as the acquisition problem."
        )
    return " ".join(notes)


def parse_json(text):
    text = text.strip()
    array_match = re.search(r"\[.*?\]", text, flags=re.S)
    if array_match:
        payload = re.sub(r"//[^\r\n]*", "", array_match.group(0))
        payload = re.sub(r",\s*([}\]])", r"\1", payload)
        obj = json.loads(payload)
        if not isinstance(obj, list) or len(obj) != len(TASKS):
            raise ValueError("Rationale array must contain exactly five strings")
        if not all(isinstance(value, str) and value.strip() for value in obj):
            raise ValueError("Rationale array entries must be non-empty strings")
        return {task: value.strip() for task, value in zip(TASKS, obj)}

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON array or object in teacher generation")
    payload = re.sub(r"//[^\r\n]*", "", match.group(0))
    payload = re.sub(r",\s*([}\]])", r"\1", payload)
    obj = json.loads(payload)
    normalized = {}
    for task in TASKS:
        value = next((obj[key] for key in EVIDENCE_ALIASES[task] if key in obj), None)
        if isinstance(value, str) and value.strip():
            normalized[task] = value.strip()
    missing = [task for task in TASKS if task not in normalized]
    if missing:
        raise ValueError(f"Missing rationale strings: {missing}")
    return normalized


def validate_rationale(rationale, states):
    """Reject state echoes so C_sem cannot silently receive pseudo-rationales."""
    normalized = {}
    for task in TASKS:
        explanation = norm(rationale.get(task)).lower().replace("_", " ")
        state = norm(states.get(task)).lower().replace("_", " ")
        normalized[task] = explanation
        if explanation == state:
            raise ValueError(f"label-only rationale for {task}: {rationale.get(task)!r}")
        task_words = task.replace("_", " ")
        decision_echoes = (
            "decision", "class label", f"{task_words} status", f"{task_words} decision",
        )
        if any(phrase in explanation for phrase in decision_echoes):
            raise ValueError(
                f"decision-label echo for {task}: {rationale.get(task)!r}"
            )

    combined = " ".join(rationale.values()).lower()
    forbidden_label_phrases = (
        "image usability", "structure visibility", "keyframe status",
        "primary defect", "recommended action", "acceptable", "unacceptable",
        "limited", "adequate", "partial", "absent", "uncertain", "candidate",
        "severe artifact", "thyroid not visible", "target incomplete",
        "repeat acquisition", "repeat the acquisition", "reacquire target",
        "adjust angle and rescan", "adjust the angle and rescan",
    )
    combined_normalized = combined.replace("_", " ")
    def contains_complete_phrase(text, phrase):
        pattern = r"(?<!\w)" + r"\s+".join(
            re.escape(part) for part in phrase.split()
        ) + r"(?!\w)"
        return re.search(pattern, text) is not None

    for phrase in forbidden_label_phrases:
        if contains_complete_phrase(combined_normalized, phrase):
            offenders = {
                task: rationale[task]
                for task in TASKS
                if contains_complete_phrase(
                    rationale[task].lower().replace("_", " "), phrase
                )
            }
            raise ValueError(
                f"rationale contains classification or task-label language: "
                f"phrase={phrase!r}, offenders={offenders!r}"
            )

    forbidden_advice = (
        "follow up", "healthcare professional", "monitor the patient", "patient's condition",
        "treatment", "tirads", "ti-rads", "diagnosis", "diagnostic use",
    )
    if any(phrase in combined for phrase in forbidden_advice):
        raise ValueError("rationale contains non-visual clinical advice")

    meta_generation = (
        "stored teacher", "teacher classification", "teacher label",
        "the prompt", "the response", "remaining evidence",
    )
    if any(phrase in combined for phrase in meta_generation):
        raise ValueError("rationale contains meta-generation language")

    cross_task_phrases = {
        "image_usability": (
            "target detection", "detector support", "thyroid detection",
            "thyroid localization", "target localization",
        ),
    }
    for task, phrases in cross_task_phrases.items():
        if any(phrase in normalized[task] for phrase in phrases):
            raise ValueError(
                f"rationale contains {task} cross-task language: {rationale[task]!r}"
            )

    semantic_focus = {
        "image_usability": (
            "clar", "bright", "contrast", "signal", "corrupt", "blur",
            "artifact", "texture", "detail", "intensity", "readable",
            "interpret", "distortion", "dropout", "dark", "gradient",
            "illuminat", "sharp", "resolv", "focus", "grayscale", "quality",
        ),
        "structure_visibility": (
            "thyroid", "gland", "parenchyma", "margin", "boundary",
            "anatom", "tissue", "target", "region", "identif", "visib",
            "localiz", "coverage",
        ),
        "keyframe_status": (
            "frame", "sequence", "sampled", "view", "stable",
            "representative", "coverage", "cross-frame", "target",
        ),
        "primary_defect": (
            "corrupt", "distort", "dropout", "coverage", "boundary",
            "margin", "field", "target", "tissue", "detector", "support",
            "loss", "omission", "truncat", "signal", "visible", "identif",
            "thyroid", "gland", "parenchyma", "localiz", "defect", "problem",
        ),
        "recommended_action": (
            "acquire", "acquisition", "additional", "new", "fresh", "restore",
            "recover", "capture", "include", "coverage", "localization",
            "field", "retain", "preserve", "needed", "sufficient",
            "frames", "reimaging", "angle",
        ),
    }
    for task, terms in semantic_focus.items():
        if not any(term in normalized[task] for term in terms):
            raise ValueError(
                f"rationale lacks {task} semantic focus: {rationale[task]!r}"
            )

    contradictions = {
        ("image_usability", "unacceptable"): (
            "clear and well-focused", "good quality", "suitable for diagnostic",
        ),
        ("structure_visibility", "uncertain"): (
            "clearly visible", "well-defined", "easily distinguishable", "fully visualized",
        ),
        ("structure_visibility", "absent"): (
            "clearly visible", "well-defined", "easily distinguishable", "fully visualized",
        ),
        ("keyframe_status", "no"): (
            "representative of the entire", "representative keyframe",
            "remains stable", "stable and complete", "stable target coverage",
        ),
        ("keyframe_status", "candidate"): (
            "stable and complete", "complete target captured", "definitive keyframe",
        ),
        ("primary_defect", "severe_artifact"): (
            "no visible defects", "no visible artifacts", "free of artifact",
        ),
        ("primary_defect", "target_incomplete"): (
            "broad streaking", "motion streaking", "severe artifact",
        ),
        ("primary_defect", "none"): (
            "broad streaking", "motion streaking", "severe artifact",
            "motion blur obscures", "corrupted field",
        ),
        ("structure_visibility", "absent"): (
            "clearly visible", "well-defined", "easily distinguishable", "fully visualized",
            "no significant motion blur",
        ),
        ("recommended_action", "repeat_acquisition"): (
            "no further action", "no additional action", "adequate as acquired",
        ),
    }
    for (task, state), phrases in contradictions.items():
        if norm(states.get(task)).lower() == state:
            explanation = normalized[task]
            if any(phrase in explanation for phrase in phrases):
                raise ValueError(f"incompatible rationale for {task}={state}")

    normalized_values = [
        re.sub(r"\W+", " ", rationale[task].lower()).strip() for task in TASKS
    ]
    if len(set(normalized_values)) != len(normalized_values):
        duplicates = {
            value: [
                task for task, candidate in zip(TASKS, normalized_values)
                if candidate == value
            ]
            for value in normalized_values
            if normalized_values.count(value) > 1
        }
        raise ValueError(f"task rationales must be distinct: {duplicates}")

    for task in TASKS:
        words = rationale_words(normalized[task])
        if len(words) < 5:
            raise ValueError(f"rationale too short for {task}: {rationale.get(task)!r}")
        if len(words) > 15:
            raise ValueError(f"rationale too long for {task}: {rationale.get(task)!r}")
    alignment_errors = semantic_alignment_errors(rationale, states)
    if alignment_errors:
        raise ValueError("semantic alignment failed: " + " | ".join(alignment_errors))
    return rationale


def compress_generated_evidence(text, task, state):
    """Remove label/meta echoes while retaining teacher-generated visual evidence."""
    value = norm(text).replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip(" \"'`-.,")
    state_words = norm(state).lower().replace("_", " ")

    # Prefer causal/evidence clauses over task-name or answer restatements.
    clauses = [
        clause.strip(" \"'`-.,")
        for clause in re.split(r"(?<=[.!?;])\s+|\s*,\s*", value)
        if clause.strip()
    ]
    meta = (
        "teacher decision", "recommended action", "decision for", "status is",
        "class label", "teacher label", "the evidence supporting the decision",
    )
    candidates = [clause for clause in clauses if not any(x in clause.lower() for x in meta)]
    if not candidates:
        candidates = clauses

    if task == "recommended_action":
        evidence_terms = (
            "coverage", "boundary", "anatom", "field", "signal", "clarity",
            "information", "detector", "frame", "target", "visibility",
        )
        preferred = [
            clause for clause in candidates
            if any(term in clause.lower() for term in evidence_terms)
        ]
        if preferred:
            candidates = preferred

    value = candidates[0] if candidates else value
    value = re.sub(
        r"^(?:for this task\s*,?\s*|based on (?:the )?visible evidence\s*,?\s*)",
        "", value, flags=re.I,
    )
    value = re.sub(
        r"\bwhich (?:may |can )?(?:suggest|indicate)(?:s)? that the .*$",
        "", value, flags=re.I,
    ).strip(" \"'`-.,")
    value = re.sub(
        r"\b(?:decision|status|class label|image usability|structure visibility|"
        r"keyframe status|primary defect|recommended action)\b",
        "", value, flags=re.I,
    )
    neutral_replacements = (
        (r"\bsevere artifacts?\b", "broad field corruption"),
        (r"\bthe thyroid (?:is|remains) not visible\b", "no identifiable thyroid tissue is present"),
        (r"\bthyroid not visible\b", "no identifiable thyroid tissue"),
        (r"\btarget incomplete\b", "missing target boundary"),
        (
            r"\b(?:a\s+)?repeat (?:the )?acquisition(?:\s+is needed)?\b",
            "new imaging is needed",
        ),
        (r"\breacquire (?:the )?target\b", "restore identifiable thyroid tissue within the field"),
        (r"\badjust (?:the )?angle and rescan\b", "recover the missing target boundary"),
        (r"^hold\b", "retain"),
        (r"\bunacceptable\b", "not reliably interpretable"),
        (r"\buncertain\b", "not reliably supported"),
        (r"\bcandidate\b", "potentially representative"),
        (r"\blimited\b", "restricted"),
    )
    for pattern, replacement in neutral_replacements:
        value = re.sub(pattern, replacement, value, flags=re.I)

    if task == "image_usability":
        value = re.sub(
            r"\s+for\s+(?:target|thyroid)\s+(?:detection|localization)\b.*$",
            "",
            value,
            flags=re.I,
        )
    if task == "primary_defect" and state_words == "thyroid not visible":
        if re.search(
            r"\b(?:thyroid|gland|parenchyma)\b.*\b(?:undetectable|not visible|"
            r"unidentifiable|not identifiable)\b|\bno identifiable thyroid\b",
            value,
            flags=re.I,
        ):
            value = "missing thyroid localization is the dominant acquisition problem"

    if state_words and state_words not in {"yes", "no", "none"}:
        value = re.sub(rf"\b{re.escape(state_words)}\b", "", value, flags=re.I)
        value = re.sub(
            rf"\b{re.escape(state_words.replace('acquisition', 'the acquisition'))}\b",
            "", value, flags=re.I,
        )
    value = re.sub(r"\s+", " ", value).strip(" \"'`-.,")

    trailing_stopwords = {
        "a", "an", "the", "to", "for", "of", "and", "or", "as", "is",
        "are", "was", "were", "with", "primary",
    }
    words = rationale_words(value)
    while words and words[-1].lower() in trailing_stopwords:
        words.pop()
    value = " ".join(words)

    words = rationale_words(value)
    if len(words) > 15:
        words = words[:15]
        while words and words[-1].lower() in {"a", "an", "the", "to", "for", "of", "and", "or", "as"}:
            words.pop()
        value = " ".join(words)
    return value


def sanitize_generated_rationale(rationale, states):
    return {
        task: compress_generated_evidence(rationale[task], task, states[task])
        for task in TASKS
    }


def select_action_evidence(rationale, states):
    """Backward-compatible alias for independently generated five-task evidence."""
    return finalize_task_rationales(rationale, states)


def finalize_task_rationales(rationale, states):
    """Preserve the teacher-generated action atom and validate all five tasks."""
    result = sanitize_generated_rationale(
        {task: norm(rationale.get(task)) for task in TASKS}, states
    )
    if any(not result[task] for task in TASKS):
        raise ValueError("all five task rationales are required")
    if result["recommended_action"].lower() == result["primary_defect"].lower():
        raise ValueError("action rationale must be generated independently")
    return validate_rationale(result, states)


def clean_plain_evidence(text, state="", task=""):
    """Normalize a rare per-task fallback while rejecting label echoes."""
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", value, flags=re.I)
    try:
        decoded = json.loads(value)
        if isinstance(decoded, str):
            value = decoded
        elif (
            isinstance(decoded, list)
            and task in TASKS
            and len(decoded) == len(TASKS)
            and all(isinstance(item, str) for item in decoded)
        ):
            # A rationale-tuned adapter may honor its learned five-item schema
            # even during a per-task repair prompt. Select only the requested
            # task instead of treating the full JSON array as one rationale.
            value = decoded[TASKS.index(task)]
        elif isinstance(decoded, dict) and task in TASKS:
            candidate = next(
                (decoded[key] for key in EVIDENCE_ALIASES[task] if key in decoded),
                None,
            )
            if isinstance(candidate, str):
                value = candidate
    except Exception:
        pass
    value = value.splitlines()[0].strip().strip('"\'` -*')
    value = re.sub(r"^(?:visible\s+)?evidence\s*:\s*", "", value, flags=re.I)
    if task:
        value = compress_generated_evidence(value, task, state)
    normalized = value.lower().replace("_", " ").strip(" .")
    state_normalized = norm(state).lower().replace("_", " ")
    if normalized == state_normalized:
        raise ValueError(f"label-only fallback evidence: {value!r}")
    if len(rationale_words(normalized)) < 5:
        raise ValueError(f"fallback evidence too short: {value!r}")
    if len(rationale_words(normalized)) > 15:
        raise ValueError(f"fallback evidence too long: {value!r}")
    return value


def fallback_prompt(
    task,
    state,
    low_level_summary=None,
    rag_entries=None,
    rag_visual_observations=None,
    prior_rationale=None,
    states=None,
):
    low_level_summary = low_level_summary or []
    rag_entries = rag_entries or []
    observations = rag_visual_observations or []
    metrics = "; ".join(task_low_level_hints(task, low_level_summary))
    retrieved_observations = "; ".join(
        render_observation_hint(item)
        for item in task_observation_hints(task, observations)
    )
    relevant_entries = [
        entry for entry in rag_entries if task in entry.get("tasks", ())
    ]
    knowledge = " ".join(
        f"[{entry['id']}] {entry['text']}" for entry in relevant_entries
    )
    focus = {
        "image_usability": "Describe only visible clarity, brightness, contrast, or corruption.",
        "structure_visibility": "Describe only visible thyroid parenchyma, margins, and field coverage.",
        "keyframe_status": "Describe only cross-frame stability and representative target coverage.",
        "primary_defect": "Describe the dominant visible acquisition problem without naming a class.",
        "recommended_action": "Describe only the acquisition objective implied by all visible evidence.",
    }[task]
    joint = ""
    if task == "recommended_action" and prior_rationale:
        joint = "\nEvidence already generated for the first four tasks:\n" + "\n".join(
            f"- {name.replace('_', ' ')}: {prior_rationale[name]}"
            for name in VISUAL_TASKS
        )
    semantic_requirements = state_condition_guidance(states or {})
    action_separation = ""
    if task == "recommended_action":
        action_separation = (
            "\nThe final phrase must express the acquisition objective and must not "
            "repeat the structure-visibility sentence."
        )
    return f"""Inspect this thyroid-ultrasound contact sheet.
The stored teacher classification for this task is {state}; explain why using evidence only.
{focus}
Objective metrics: {metrics}
Archived sample observations: {retrieved_observations}
Relevant QC knowledge: {knowledge}
{joint}
State-specific requirements: {semantic_requirements}
{action_separation}
Return only 5-15 words of sample-specific visual evidence.
Do not state a task name, class, decision, diagnosis, or treatment advice.
Return plain evidence text only."""


@torch.no_grad()
def generate_visual_fallback(model, processor, item):
    """Generate task-local evidence, then condition Action on the first four atoms."""
    texts, images = [], []
    for task in VISUAL_TASKS:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": fallback_prompt(
                    task,
                    item["states"][task],
                    item["low_level_summary"],
                    item["rag_entries"],
                    item["rag_visual_observations"],
                    states=item["states"],
                )},
            ],
        }]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        images.append(Image.open(item["image_path"]).convert("RGB"))
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {
        key: value.to("cuda:0") if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    generated = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        use_cache=True,
        stop_strings=["\n"],
        tokenizer=processor.tokenizer,
    )
    answer_ids = generated[:, inputs["input_ids"].shape[1]:]
    raw_values = processor.batch_decode(answer_ids, skip_special_tokens=True)
    rationale = {
        task: clean_plain_evidence(raw, item["states"][task], task)
        for task, raw in zip(VISUAL_TASKS, raw_values)
    }

    action_messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": fallback_prompt(
                "recommended_action",
                item["states"]["recommended_action"],
                item["low_level_summary"],
                item["rag_entries"],
                item["rag_visual_observations"],
                rationale,
                item["states"],
            )},
        ],
    }]
    action_text = processor.apply_chat_template(
        action_messages, tokenize=False, add_generation_prompt=True
    )
    action_inputs = processor(
        text=[action_text],
        images=[Image.open(item["image_path"]).convert("RGB")],
        padding=True,
        return_tensors="pt",
    )
    action_inputs = {
        key: value.to("cuda:0") if hasattr(value, "to") else value
        for key, value in action_inputs.items()
    }
    action_generated = model.generate(
        **action_inputs,
        max_new_tokens=40,
        do_sample=False,
        use_cache=True,
        stop_strings=["\n"],
        tokenizer=processor.tokenizer,
    )
    action_answer = processor.batch_decode(
        action_generated[:, action_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    rationale["recommended_action"] = clean_plain_evidence(
        action_answer,
        item["states"]["recommended_action"],
        "recommended_action",
    )
    return finalize_task_rationales(rationale, item["states"]), raw_values + [action_answer]


def build_inputs(processor, items, corrective_notes=None):
    corrective_notes = corrective_notes or [""] * len(items)
    texts, images = [], []
    for item, note in zip(items, corrective_notes):
        instruction = prompt(
            item["states"],
            item["low_level_summary"],
            item["rag_entries"],
            item["rag_visual_observations"],
        )
        if note:
            control_guidance = state_aware_retry_guidance(item["states"])
            instruction += (
                "\nYour previous response was rejected because: " + note +
                "\n" + control_guidance +
                "\nRegenerate all five JSON strings with concrete visible evidence only."
            )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        images.append(Image.open(item["image_path"]).convert("RGB"))
    return processor(text=texts, images=images, padding=True, return_tensors="pt")


@torch.no_grad()
def generate_batch(model, processor, items, max_new_tokens, corrective_notes=None):
    inputs = build_inputs(processor, items, corrective_notes)
    inputs = {
        key: value.to("cuda:0") if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        stop_strings=["]"],
        tokenizer=processor.tokenizer,
    )
    answer_ids = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(answer_ids, skip_special_tokens=True)


def failed_generation_record(item, error, raw=""):
    """Preserve a failed generation as an auditable, retryable record.

    This is deliberately distinct from a teacher/RAG conflict.  It must never
    contribute to G2D-UOT, and a later invocation removes and retries it.
    """
    return {
        "sample_id": item["sample_id"],
        "segment_id": item["segment_id"],
        "review_id": item["review_id"],
        "image_path": item["image_path"],
        "teacher_states": item["states"],
        "teacher_probabilities": item["probabilities"],
        "state_source": item["state_source"],
        "structured_rationale": None,
        "teacher_knowledge_usable": False,
        "teacher_knowledge_conflicts": [],
        "rationale_generation_error": str(error),
        "low_level_information": item["low_level_information"],
        "low_level_summary": item["low_level_summary"],
        "rag_visual_observations": item["rag_visual_observations"],
        "rag_computed_metrics": item["rag_computed_metrics"],
        "retrieved_rule_ids": [entry["id"] for entry in item["rag_entries"]],
        "rag_source_ids": sorted({
            source
            for entry in item["rag_entries"]
            for source in entry["source_ids"]
        } | set(item["historical_rag_source_ids"])),
        "rag_version": "legacy_thyqc_rag_v2_soft_action_20260924",
        "rag_provenance": RAG_PROVENANCE,
        "rationale_derivation": "generation_failed_pending_recovery",
        "teacher_generation_raw": raw,
    }


def low_margin_failure_conflicts(item, error, margin_threshold=0.01):
    """Mask semantic transport only when the failed task is probability-ambiguous."""
    message = str(error or "")
    conflicts = []
    for task in TASKS:
        if f"{task}:" not in message:
            continue
        distribution = item.get("probabilities", {}).get(task, {})
        ranked = sorted((float(value) for value in distribution.values()), reverse=True)
        if len(ranked) < 2:
            continue
        margin = ranked[0] - ranked[1]
        if margin <= margin_threshold:
            conflicts.append(f"low_margin_semantic_ambiguity:{task}:{margin:.6f}")
    return conflicts


def semantic_ambiguity_record(item, error, conflicts, raw=""):
    """Preserve labels/probabilities while permanently masking ambiguous C_sem."""
    record = failed_generation_record(item, error, raw)
    record.pop("rationale_generation_error", None)
    record["teacher_knowledge_conflicts"] = list(conflicts)
    record["semantic_ambiguity_error"] = str(error)
    record["rationale_derivation"] = "masked_low_margin_semantic_ambiguity"
    return record


def terminal_quality_mask_record(item, error, raw=""):
    """Retain all supervision while excluding irreparable rationale cross-talk."""
    record = failed_generation_record(item, error, raw)
    record.pop("rationale_generation_error", None)
    record["teacher_knowledge_conflicts"] = [
        "rationale_generation_quality_failure"
    ]
    record["rationale_quality_error"] = str(error)
    record["rationale_derivation"] = "masked_rationale_generation_quality_failure"
    return record


def remove_retryable_generation_failures(out):
    """Atomically remove prior generation failures so only they are retried."""
    out = Path(out)
    if not out.exists():
        return []
    rows = list(read_jsonl(out))
    failed_ids = [
        row.get("sample_id") for row in rows
        if row.get("rationale_generation_error")
    ]
    if not failed_ids:
        return []
    retained = [row for row in rows if not row.get("rationale_generation_error")]
    temporary = out.with_name(out.name + ".recovery.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(out)
    return failed_ids


def reconcile_existing_records(out, request_index):
    """Keep only records that remain valid under the current semantic policy.

    Existing output is a resumable cache, not an authority.  Whenever the
    state-conditioned semantic rules change, stale masks, failed generations,
    state drift, and semantically invalid rationales are removed atomically so
    that the normal generation loop retries exactly those samples.
    """
    out = Path(out)
    summary = {
        "kept_valid": 0,
        "kept_current_mask": 0,
        "removed_generation_failure": 0,
        "removed_semantic_invalid": 0,
        "removed_stale_mask": 0,
        "removed_state_drift": 0,
        "removed_unknown_sample": 0,
    }
    if not out.exists():
        return summary

    rows = list(read_jsonl(out))
    retained = []
    for record in rows:
        sample_id = record.get("sample_id")
        item = request_index.get(sample_id)
        if item is None:
            summary["removed_unknown_sample"] += 1
            continue
        if record.get("rationale_generation_error"):
            summary["removed_generation_failure"] += 1
            continue

        current_states = item["states"]
        recorded_states = record.get("teacher_states") or current_states
        if any(norm(recorded_states.get(task)) != norm(current_states.get(task)) for task in TASKS):
            summary["removed_state_drift"] += 1
            continue

        current_usable = bool(item.get("teacher_knowledge_usable", True))
        recorded_usable = bool(record.get("teacher_knowledge_usable", True))
        if not current_usable:
            current_conflicts = sorted(item.get("teacher_knowledge_conflicts") or [])
            recorded_conflicts = sorted(record.get("teacher_knowledge_conflicts") or [])
            if not recorded_usable and record.get("structured_rationale") is None and recorded_conflicts == current_conflicts:
                retained.append(record)
                summary["kept_current_mask"] += 1
            else:
                summary["removed_stale_mask"] += 1
            continue

        if not recorded_usable or record.get("structured_rationale") is None:
            ambiguity_error = record.get("semantic_ambiguity_error")
            current_conflicts = low_margin_failure_conflicts(item, ambiguity_error)
            recorded_conflicts = record.get("teacher_knowledge_conflicts") or []
            if (
                ambiguity_error
                and sorted(current_conflicts) == sorted(recorded_conflicts)
                and current_conflicts
            ):
                retained.append(record)
                summary["kept_current_mask"] += 1
            elif (
                record.get("rationale_quality_error")
                and recorded_conflicts == ["rationale_generation_quality_failure"]
            ):
                retained.append(record)
                summary["kept_current_mask"] += 1
            else:
                summary["removed_stale_mask"] += 1
            continue
        try:
            validate_rationale(record["structured_rationale"], current_states)
        except Exception:
            summary["removed_semantic_invalid"] += 1
            continue
        retained.append(record)
        summary["kept_valid"] += 1

    if retained != rows:
        backup = out.with_name(out.name + ".pre_semantic_reconcile.bak")
        if not backup.exists():
            with backup.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary = out.with_name(out.name + ".reconcile.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in retained:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(out)
    return summary


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument(
        "--adapter-dir",
        required=True,
        help="Fine-tuned LLaVA-Med teacher LoRA used for rationale generation.",
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher-jsonl", required=True)
    ap.add_argument(
        "--rag-evidence-jsonl",
        default="",
        help="Original ThyQC sample-level RAG evidence manifest.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--generation-seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)

    requests = unique_requests(
        args.manifest, args.teacher_jsonl, args.rag_evidence_jsonl
    )
    if args.limit:
        requests = requests[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    request_index = {row["sample_id"]: row for row in requests}
    reconciliation = reconcile_existing_records(out, request_index)
    if any(value for key, value in reconciliation.items() if key.startswith("removed_")):
        print(json.dumps({
            "event": "existing_rationale_cache_reconciled",
            **reconciliation,
        }, ensure_ascii=False), flush=True)
    done = set()
    if out.exists():
        done = {row["sample_id"] for row in read_jsonl(out)}
    requests = [row for row in requests if row["sample_id"] not in done]

    # Do not ask the generator to invent a rationale for a contradictory
    # teacher package.  Preserve the record for auditing and let Stage I mask
    # only its G2D-UOT term; L_cls and L_prob still use the sample.
    unusable = [row for row in requests if not row["teacher_knowledge_usable"]]
    if unusable:
        with out.open("a", encoding="utf-8") as handle:
            for item in unusable:
                record = {
                    "sample_id": item["sample_id"],
                    "segment_id": item["segment_id"],
                    "review_id": item["review_id"],
                    "image_path": item["image_path"],
                    "teacher_states": item["states"],
                    "teacher_probabilities": item["probabilities"],
                    "state_source": item["state_source"],
                    "structured_rationale": None,
                    "teacher_knowledge_usable": False,
                    "teacher_knowledge_conflicts": item["teacher_knowledge_conflicts"],
                    "low_level_information": item["low_level_information"],
                    "low_level_summary": item["low_level_summary"],
                    "rag_visual_observations": item["rag_visual_observations"],
                    "rag_computed_metrics": item["rag_computed_metrics"],
                    "retrieved_rule_ids": [entry["id"] for entry in item["rag_entries"]],
                    "rag_source_ids": sorted({
                        source
                        for entry in item["rag_entries"]
                        for source in entry["source_ids"]
                    } | set(item["historical_rag_source_ids"])),
                    "rag_version": "legacy_thyqc_rag_v2_soft_action_20260924",
                    "rag_provenance": RAG_PROVENANCE,
                    "rationale_derivation": "masked_teacher_package_conflict",
                    "teacher_generation_raw": "",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "event": "teacher_knowledge_conflicts_masked",
            "records": len(unusable),
            "reason_counts": {
                reason: sum(reason in row["teacher_knowledge_conflicts"] for row in unusable)
                for reason in sorted({
                    reason for row in unusable
                    for reason in row["teacher_knowledge_conflicts"]
                })
            },
        }), flush=True)
    requests = [row for row in requests if row["teacher_knowledge_usable"]]

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    processor = patch_processor(
        AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True), config
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_dir,
        quantization_config=quant,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    started = time.time()
    completed = 0
    generation_failures = 0
    with out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(requests), args.batch_size):
            items = requests[start:start + args.batch_size]
            raws = [text.strip() for text in generate_batch(
                model, processor, items, args.max_new_tokens
            )]
            parsed = [None] * len(items)
            errors = [""] * len(items)
            for index, (item, raw) in enumerate(zip(items, raws)):
                try:
                    parsed[index] = finalize_task_rationales(
                        parse_json(raw), item["states"]
                    )
                except Exception as exc:
                    errors[index] = str(exc)

            for attempt in range(args.max_retries):
                failed = [index for index, value in enumerate(parsed) if value is None]
                if not failed:
                    break
                retry_items = [items[index] for index in failed]
                retry_notes = [errors[index] for index in failed]
                retry_raws = generate_batch(
                    model, processor, retry_items, args.max_new_tokens, retry_notes
                )
                for index, retry_raw in zip(failed, retry_raws):
                    raws[index] = retry_raw.strip()
                    try:
                        parsed[index] = finalize_task_rationales(
                            parse_json(raws[index]), items[index]["states"]
                        )
                    except Exception as exc:
                        errors[index] = str(exc)

            failed = [index for index, value in enumerate(parsed) if value is None]
            if failed:
                for index in failed:
                    try:
                        fallback, fallback_raw = generate_visual_fallback(
                            model, processor, items[index]
                        )
                        parsed[index] = fallback
                        raws[index] = json.dumps({
                            "structured_attempt": raws[index],
                            "fallback_visual_generation": fallback_raw,
                        }, ensure_ascii=False)
                        print(json.dumps({
                            "event": "fallback_visual_generation",
                            "sample_id": items[index]["sample_id"],
                            "structured_error": errors[index],
                        }), flush=True)
                    except Exception as exc:
                        errors[index] = f"fallback failed: {exc}"

            failed = [index for index, value in enumerate(parsed) if value is None]
            if failed:
                for index in failed:
                    conflicts = low_margin_failure_conflicts(
                        items[index], errors[index]
                    )
                    if conflicts:
                        record = semantic_ambiguity_record(
                            items[index], errors[index], conflicts, raws[index]
                        )
                        event = "rationale_low_margin_ambiguity_masked"
                    else:
                        record = terminal_quality_mask_record(
                            items[index], errors[index], raws[index]
                        )
                        event = "rationale_quality_failure_masked"
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed += 1
                    print(json.dumps({
                        "event": event,
                        "sample_id": items[index]["sample_id"],
                        "error": errors[index],
                        "conflicts": conflicts,
                    }, ensure_ascii=False), flush=True)

            for item, raw, rationale in zip(items, raws, parsed):
                if rationale is None:
                    continue
                record = {
                    "sample_id": item["sample_id"],
                    "segment_id": item["segment_id"],
                    "review_id": item["review_id"],
                    "image_path": item["image_path"],
                    "teacher_states": item["states"],
                    "teacher_probabilities": item["probabilities"],
                    "state_source": item["state_source"],
                    "structured_rationale": rationale,
                    "low_level_information": item["low_level_information"],
                    "low_level_summary": item["low_level_summary"],
                    "rag_visual_observations": item["rag_visual_observations"],
                    "rag_computed_metrics": item["rag_computed_metrics"],
                    "retrieved_rule_ids": [entry["id"] for entry in item["rag_entries"]],
                    "rag_source_ids": sorted({
                        source
                        for entry in item["rag_entries"]
                        for source in entry["source_ids"]
                    } | set(item["historical_rag_source_ids"])),
                    "rag_version": "legacy_thyqc_rag_v2_soft_action_20260924",
                    "rag_provenance": RAG_PROVENANCE,
                    "rationale_derivation": "teacher_generated_label_debiased_short_evidence",
                    "teacher_knowledge_usable": True,
                    "teacher_knowledge_conflicts": [],
                    "teacher_generation_raw": raw,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1
            handle.flush()
            if completed == len(items) or completed % 25 < len(items) or completed == len(requests):
                elapsed = time.time() - started
                print(json.dumps({
                    "event": "rationale_progress",
                    "new_records": completed,
                    "remaining": len(requests) - completed,
                    "sec_per_sample": elapsed / completed,
                    "batch_size": args.batch_size,
                }), flush=True)

    if generation_failures:
        raise RuntimeError(
            f"Deferred {generation_failures} rationale generation failures; "
            "rerun to atomically remove and retry only those records."
        )


if __name__ == "__main__":
    main()
