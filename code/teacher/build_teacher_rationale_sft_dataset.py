"""Build sample-specific rationale-generation supervision from the original ThyQC RAG."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from generate_teacher_structured_rationales import (
    TASKS,
    RAG_KNOWLEDGE,
    derive_rag_visual_observations,
    low_level_information,
    prompt,
    rag_evidence_index,
    rationale_words,
    retrieve_rag_entries,
    summarize_low_level_information,
    validate_rationale,
)


DOCTOR_COLUMNS = {task: f"doctor_{task}" for task in TASKS}
ACTION_LABELS = {
    "hold",
    "repeat_acquisition",
    "reacquire_target",
    "adjust_angle_and_rescan",
}


def resolve_adjudicated_states(source, action_review=None):
    """Keep four physician labels immutable and apply only approved action edits.

    The original physician row remains the authority.  An action review is an
    auditable overlay: pending suggestions do not change supervision, and an
    approved suggestion must identify the original action it reviewed.
    """
    states = {task: source[DOCTOR_COLUMNS[task]] for task in TASKS}
    review = action_review or {}
    forbidden = [
        f"revised_{task}"
        for task in TASKS[:-1]
        if str(review.get(f"revised_{task}") or "").strip()
    ]
    if forbidden:
        raise ValueError(
            "Only recommended_action may be revised; found " + ", ".join(forbidden)
        )

    provenance = {
        "review_status": str(review.get("review_status") or "not_reviewed"),
        "changed_fields": [],
        "original_recommended_action": states["recommended_action"],
        "reviewer": str(review.get("reviewer") or ""),
        "review_evidence": str(review.get("evidence") or ""),
    }
    if provenance["review_status"] != "approved":
        return states, provenance

    reviewed_original = str(review.get("original_action") or "").strip()
    if reviewed_original != states["recommended_action"]:
        raise ValueError(
            f"original action mismatch for {source.get('sample_id')}: "
            f"review={reviewed_original!r}, source={states['recommended_action']!r}"
        )
    revised = str(review.get("revised_action") or "").strip()
    if revised not in ACTION_LABELS:
        raise ValueError(f"invalid revised action for {source.get('sample_id')}: {revised!r}")
    if revised != states["recommended_action"]:
        states["recommended_action"] = revised
        provenance["changed_fields"] = ["recommended_action"]
    return states, provenance


def action_review_index(path):
    if not path:
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    index = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("action review row is missing sample_id")
        if sample_id in index:
            raise ValueError(f"duplicate action review for {sample_id}")
        index[sample_id] = row
    return index


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_choice(sample_id, task, values):
    digest = hashlib.sha256(f"{sample_id}|{task}".encode()).digest()
    return values[int.from_bytes(digest[:4], "big") % len(values)]


def derived_observations(states, metrics):
    defect = states["primary_defect"]
    if defect == "severe_artifact":
        observations = ["severe_artifact_pattern", "anatomy_not_assessable"]
        if metrics.get("bright_ratio", 0) > 0.20 or metrics.get("dark_ratio", 0) > 0.35:
            observations.append("no_signal_or_bright_screen")
        return observations
    if defect == "target_incomplete":
        return ["thyroid_region_partly_visible", "incomplete_target_coverage"]
    if defect == "thyroid_not_visible":
        return ["no_reliable_thyroid_parenchyma", "weak_or_absent_yolo_support"]
    return ["thyroid_region_visible", "target_complete", "usable_keyframe"]


def metric_descriptor(metrics, sample_id, task):
    brightness = float(metrics.get("brightness_mean", 0.5))
    contrast = float(metrics.get("contrast_std", 0.2))
    gradient = float(metrics.get("gradient_mean", 0.07))
    det_count = int(metrics.get("det_count", 0) or 0)
    max_conf = float(metrics.get("max_conf", 0.0) or 0.0)
    visual_choices = []
    if brightness < 0.30:
        visual_choices += ["relatively dark field", "reduced overall brightness"]
    elif brightness > 0.68:
        visual_choices += ["bright-dominant field", "elevated overall brightness"]
    else:
        visual_choices += ["mid-range brightness", "balanced overall brightness"]
    if contrast < 0.16:
        visual_choices += ["narrow grayscale contrast", "weak intensity separation"]
    elif contrast > 0.29:
        visual_choices += ["strong intensity variation", "broad grayscale variation"]
    else:
        visual_choices += ["moderate grayscale contrast", "distributed tissue contrast"]
    if gradient < 0.035:
        visual_choices += ["weak local gradients", "soft local edge definition"]
    elif gradient > 0.11:
        visual_choices += ["pronounced local edges", "strong local edge variation"]
    else:
        visual_choices += ["moderate local gradients", "preserved local edge variation"]

    detector_choices = []
    if det_count:
        strength = "strong" if max_conf >= 0.70 else "weak"
        detector_choices += [
            f"{strength} detector support across {det_count} frames",
            f"thyroid-region support in {det_count} sampled frames",
        ]
    else:
        detector_choices += [
            "no detector-supported thyroid region",
            "missing detector support across sampled frames",
        ]

    if task == "image_usability":
        choices = visual_choices
    elif task in {"structure_visibility", "keyframe_status"}:
        choices = detector_choices
    elif task == "primary_defect":
        choices = visual_choices + detector_choices
    else:
        choices = detector_choices + visual_choices
    return stable_choice(sample_id, task, choices)


RATIONALE_BANK = {
    "image_usability": {
        "acceptable": [
            "Tissue texture remains discernible with {metric}",
            "Fine anatomical detail remains readable under {metric}",
            "The sampled frames preserve interpretable tissue detail and {metric}",
            "Visible anatomy retains coherent texture despite {metric}",
        ],
        "limited": [
            "Fine tissue detail is reduced by {metric}",
            "Uneven image characteristics weaken detail while anatomy remains interpretable",
            "Frame quality varies and {metric} reduces fine-detail confidence",
            "Anatomy remains readable although {metric} lowers visual clarity",
        ],
        "unacceptable": [
            "Display corruption and {metric} obscure assessable tissue detail",
            "Severe field distortion prevents reliable anatomical interpretation across frames",
            "Signal failure dominates the contact sheet and hides tissue detail",
            "Broad visual corruption overwhelms anatomy despite {metric}",
        ],
    },
    "structure_visibility": {
        "adequate": [
            "Thyroid parenchyma and margins remain traceable across sampled frames",
            "Target boundaries and surrounding context remain continuously represented",
            "Reliable thyroid tissue is visible with complete field coverage",
            "Gland margins remain identifiable across the sampled sequence",
        ],
        "partial": [
            "A lateral thyroid boundary exits the field in several frames",
            "Only part of the thyroid extent remains traceable across frames",
            "Visible parenchyma persists while one target margin is cut off",
            "Target tissue is present but field coverage misses a boundary",
        ],
        "absent": [
            "No reliable thyroid parenchyma or margin is identifiable across frames",
            "The imaged field lacks recognizable thyroid tissue and boundary context",
            "Expected gland texture cannot be separated from surrounding structures",
            "No detector-supported thyroid region can be confirmed in the field",
        ],
        "uncertain": [
            "Variable signal and {metric} prevent confident thyroid margin tracing",
            "Anatomical support is inconsistent and target boundaries remain unreliable",
            "Field corruption obscures the tissue needed for confident localization",
            "Weak regional support prevents consistent gland identification across frames",
        ],
    },
    "keyframe_status": {
        "yes": [
            "A stable frame preserves complete target coverage and surrounding context",
            "One sampled frame clearly represents the full thyroid field",
            "Cross-frame stability supports a representative view with complete boundaries",
            "The sequence contains a stable view retaining the entire target",
        ],
        "candidate": [
            "Target tissue remains visible but coverage varies across sampled frames",
            "A potentially representative frame still omits part of one boundary",
            "Frame stability is present while target completeness remains inconsistent",
            "The best sampled view retains anatomy but lacks complete field context",
        ],
        "no": [
            "No sampled frame retains stable anatomy with representative target coverage",
            "Cross-frame corruption prevents selection of a reliable representative view",
            "The sequence lacks a frame combining stable signal and complete context",
            "No frame consistently preserves identifiable thyroid tissue and boundaries",
        ],
    },
    "primary_defect": {
        "none": [
            "No dominant corruption or coverage loss is evident across frames",
            "The sequence shows no major acquisition problem affecting target assessment",
            "Target coverage and signal remain free of a dominant visual failure",
            "No single image defect materially obscures the thyroid field",
        ],
        "severe_artifact": [
            "Signal dropout and field corruption dominate the visible anatomy",
            "Broad display distortion obscures tissue across the sampled frames",
            "Severe visual corruption is the main limitation within the field",
            "Field-wide signal failure prevents reliable thyroid assessment",
        ],
        "thyroid_not_visible": [
            "Expected thyroid parenchyma is not identifiable within the imaged region",
            "The field shows no reliable gland texture or supporting margins",
            "Visible anatomy does not contain a confirmable thyroid target",
            "Lack of regional support prevents localization of thyroid tissue",
        ],
        "target_incomplete": [
            "A thyroid boundary remains cut off at the field edge",
            "Missing lateral coverage is the dominant acquisition limitation",
            "The visible target is truncated before its full extent appears",
            "Incomplete field coverage omits part of the thyroid margin",
        ],
    },
    "recommended_action": {
        "hold": [
            "Current stable coverage can be retained without additional image acquisition",
            "Existing frames already preserve the necessary target and field context",
            "The acquired sequence provides sufficient anatomy for continued quality review",
            "No additional coverage is needed to preserve interpretable thyroid evidence",
        ],
        "repeat_acquisition": [
            "Fresh imaging should restore interpretable signal and stable anatomical coverage",
            "A new acquisition should replace the corrupted field with readable anatomy",
            "Reimaging should recover stable signal before target coverage is reassessed",
            "New frames are needed to restore interpretable tissue and boundary evidence",
        ],
        "reacquire_target": [
            "New target localization should place identifiable thyroid tissue within the field",
            "Fresh localization is needed to recover gland tissue and surrounding context",
            "A new field should capture recognizable thyroid parenchyma and margins",
            "Targeted reacquisition should restore a confirmable thyroid region",
        ],
        "adjust_angle_and_rescan": [
            "Additional angled coverage should recover the missing thyroid boundary",
            "A changed viewing angle should include the omitted target extent",
            "Expanded field coverage is needed to restore the truncated margin",
            "Angle adjustment should capture the missing lateral thyroid context",
        ],
    },
}


def build_rationale(sample_id, states, metrics):
    rationale = {}
    for task in TASKS:
        template = stable_choice(sample_id, task, RATIONALE_BANK[task][states[task]])
        metric = metric_descriptor(metrics, sample_id, task)
        rationale[task] = template.format(metric=metric)
    return validate_rationale(rationale, states)


def resolve_rag_supervision(states, metrics, archived=None):
    """Prefer the original sample-level ThyQC RAG record; derive only if absent."""
    archived = archived or {}
    observations = list(archived.get("visual_observation") or [])
    if not observations:
        observations = derive_rag_visual_observations(states, metrics)
    archived_rule_ids = list(archived.get("retrieved_rule_ids") or [])
    rag_entries = retrieve_rag_entries(states, metrics, archived_rule_ids)
    rule_ids = [entry["id"] for entry in rag_entries]
    source_ids = list(archived.get("source_ids") or [])
    if not source_ids:
        source_ids = sorted({
            source_id
            for entry in rag_entries
            for source_id in entry.get("source_ids", [])
        })
    return observations, rule_ids, source_ids


def manifest_index(path):
    index = {}
    for row in read_jsonl(path):
        for key in ("review_id", "sample_id", "segment_id"):
            if row.get(key):
                index[str(row[key])] = row
    return index


def build_rows(csv_path, manifest_path, rag_evidence_path="", action_review_path=""):
    manifest = manifest_index(manifest_path)
    historical_rag = rag_evidence_index(rag_evidence_path)
    action_reviews = action_review_index(action_review_path)
    rag_by_id = {entry["id"]: entry for entry in RAG_KNOWLEDGE}
    rows = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            image_row = manifest.get(source["review_id"]) or manifest.get(source["sample_id"])
            if not image_row:
                raise KeyError(f"missing image for {source['review_id']}")
            image_path = image_row.get("contact_sheet") or source.get("contact_sheet")
            states, label_provenance = resolve_adjudicated_states(
                source, action_reviews.get(source["sample_id"])
            )
            metrics = low_level_information(image_path, image_row.get("auxiliary_context", ""))
            aliases = (
                source.get("review_id"), source.get("sample_id"),
                image_row.get("segment_id"), image_row.get("sample_id"),
            )
            archived = next(
                (historical_rag[str(alias)] for alias in aliases if alias and str(alias) in historical_rag),
                {},
            )
            observations, rule_ids, source_ids = resolve_rag_supervision(
                states, metrics, archived
            )
            rag_entries = [rag_by_id[rule_id] for rule_id in rule_ids if rule_id in rag_by_id]
            low_summary = summarize_low_level_information(metrics)
            sample_id = source["sample_id"]
            rationale = build_rationale(sample_id, states, metrics)
            rows.append({
                "review_id": source["review_id"],
                "sample_id": sample_id,
                "image_path": image_path,
                "teacher_states": states,
                "label_provenance": label_provenance,
                "low_level_information": metrics,
                "low_level_summary": low_summary,
                "rag_visual_observations": observations,
                "retrieved_rule_ids": rule_ids,
                "rag_source_ids": source_ids,
                "user_prompt": prompt(states, low_summary, rag_entries, observations),
                "target_rationale": rationale,
                "target_json": json.dumps([rationale[task] for task in TASKS], ensure_ascii=False),
                "supervision_provenance": (
                    "archived_original_thyqc_rag_plus_objective_visual_metrics"
                    if archived else
                    "original_thyqc_rag_vocabulary_fallback_plus_objective_visual_metrics"
                ),
            })
    return rows


def audit(rows):
    task_counts = {task: Counter(row["target_rationale"][task] for row in rows) for task in TASKS}
    return {
        "n": len(rows),
        "unique_per_task": {task: len(task_counts[task]) for task in TASKS},
        "max_duplicate_fraction_per_task": {
            task: max(task_counts[task].values()) / max(1, len(rows)) for task in TASKS
        },
        "word_count_range_per_task": {
            task: [
                min(len(rationale_words(row["target_rationale"][task])) for row in rows),
                max(len(rationale_words(row["target_rationale"][task])) for row in rows),
            ]
            for task in TASKS
        },
        "unique_full_packages": len({row["target_json"] for row in rows}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rag-evidence-jsonl", default="")
    parser.add_argument("--action-review-csv", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", required=True)
    args = parser.parse_args()
    rows = build_rows(
        args.csv,
        args.manifest,
        args.rag_evidence_jsonl,
        args.action_review_csv,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = audit(rows)
    Path(args.audit_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
