"""Audit rationale-aware Teacher Knowledge Packages for ThyQC G2D-UOT."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rationale_semantic_rules import semantic_alignment_errors


TASKS = (
    "image_usability",
    "structure_visibility",
    "keyframe_status",
    "primary_defect",
    "recommended_action",
)

FORBIDDEN_LABEL_WORDS = {
    "acceptable", "unacceptable", "limited", "adequate", "partial",
    "absent", "uncertain", "candidate", "severe_artifact",
    "thyroid_not_visible", "target_incomplete", "repeat_acquisition",
    "reacquire_target", "adjust_angle_and_rescan",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rationale_words(value):
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", str(value or ""))


def norm_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def find_label_echoes(value):
    """Find exact label words/phrases without substring false positives."""
    text = norm_text(value).replace("-", " ").replace("_", " ")
    return sorted(
        label for label in FORBIDDEN_LABEL_WORDS
        if re.search(
            rf"(?<![a-z0-9]){re.escape(label.replace('_', ' '))}(?![a-z0-9])",
            text,
        )
    )


def audit_record(record):
    """Return a flat, human-reviewable audit row for one package."""
    errors = []
    usable = bool(record.get("teacher_knowledge_usable", True))
    conflicts = list(record.get("teacher_knowledge_conflicts") or [])
    rationales = record.get("structured_rationale")
    provenance = record.get("rag_provenance") or {}
    rules = list(record.get("retrieved_rule_ids") or [])
    states = record.get("teacher_states") or {}

    if provenance.get("source_variant") != "A3_prompt_checklist_rag":
        errors.append("wrong_rag_source_variant")
    if provenance.get("original_rule_count") != 7:
        errors.append("wrong_original_rag_rule_count")
    if not rules:
        errors.append("missing_retrieved_rule_ids")

    generation_error = str(record.get("rationale_generation_error") or "").strip()
    if generation_error:
        if usable:
            errors.append("generation_failure_marked_usable")
        if conflicts:
            errors.append("generation_failure_claims_teacher_conflict")
        if rationales is not None:
            errors.append("generation_failure_has_rationale")
        status = "generation_failed" if not errors else "invalid_generation_failure"
    elif not usable:
        if not conflicts:
            errors.append("masked_without_conflict_reason")
        if rationales is not None:
            errors.append("masked_record_has_rationale")
        status = "masked_conflict" if not errors else "invalid_masked_record"
    else:
        if not isinstance(rationales, dict):
            errors.append("missing_structured_rationale")
        else:
            missing = [task for task in TASKS if not str(rationales.get(task, "")).strip()]
            if missing:
                errors.append("missing_tasks:" + ",".join(missing))
            normalized = []
            for task in TASKS:
                text = str(rationales.get(task, "")).strip()
                count = len(rationale_words(text))
                if text and not 5 <= count <= 15:
                    errors.append(f"word_count_{task}:{count}")
                echoes = find_label_echoes(text)
                if echoes:
                    errors.append(f"label_echo_{task}:" + ",".join(echoes))
                normalized.append(norm_text(text))
            duplicates = [text for text, count in Counter(normalized).items() if text and count > 1]
            if duplicates:
                errors.append("duplicate_task_rationales")
            for error in semantic_alignment_errors(rationales, states):
                errors.append("semantic_alignment:" + error)
        if not record.get("teacher_probabilities"):
            errors.append("missing_teacher_probabilities")
        status = "usable_valid" if not errors else "invalid_generated_package"

    row = {
        "sample_id": record.get("sample_id", ""),
        "segment_id": record.get("segment_id", ""),
        "review_id": record.get("review_id", ""),
        "image_path": record.get("image_path", ""),
        "audit_status": status,
        "audit_errors": errors,
        "teacher_states": states,
        "structured_rationale": rationales,
        "retrieved_rule_ids": rules,
        "rag_visual_observations": record.get("rag_visual_observations") or [],
        "low_level_summary": record.get("low_level_summary") or [],
        "teacher_knowledge_conflicts": conflicts,
        "rationale_generation_error": generation_error,
        "rag_provenance": provenance,
    }
    return row


def stratified_manual_sample(records, per_defect=8, conflict_limit=20):
    """Select deterministic class-stratified valid cases plus conflict cases."""
    selected = []
    by_defect = defaultdict(list)
    conflicts = []
    failures = []
    for row in records:
        if row.get("audit_status") == "usable_valid":
            defect = (row.get("teacher_states") or {}).get("primary_defect", "unknown")
            by_defect[defect].append(row)
        elif row.get("audit_status") == "masked_conflict":
            conflicts.append(row)
        elif row.get("audit_status") in {
            "generation_failed", "invalid_generation_failure",
            "invalid_generated_package", "invalid_masked_record",
        }:
            failures.append(row)
    for defect in ("none", "severe_artifact", "thyroid_not_visible", "target_incomplete"):
        selected.extend(sorted(by_defect.get(defect, []), key=lambda row: row["sample_id"])[:per_defect])
    selected.extend(sorted(conflicts, key=lambda row: row["sample_id"])[:conflict_limit])
    selected.extend(sorted(failures, key=lambda row: row["sample_id"]))
    return selected


def manifest_image_index(path):
    index = {}
    if not path:
        return index
    for row in read_jsonl(path):
        paths = row.get("input_paths") or [row.get("input_path") or row.get("contact_sheet")]
        image_path = next((str(value) for value in paths if value), "")
        for key in ("original_sample_id", "segment_id", "review_id", "sample_id"):
            value = str(row.get(key) or "").strip()
            if value and image_path:
                index[value] = image_path
    return index


def flatten_for_csv(row):
    flat = {
        "sample_id": row["sample_id"],
        "segment_id": row["segment_id"],
        "review_id": row["review_id"],
        "image_path": row["image_path"],
        "audit_status": row["audit_status"],
        "audit_errors": " | ".join(row["audit_errors"]),
        "retrieved_rule_ids": " | ".join(row["retrieved_rule_ids"]),
        "rag_visual_observations": " | ".join(row["rag_visual_observations"]),
        "low_level_summary": " | ".join(row["low_level_summary"]),
        "teacher_knowledge_conflicts": " | ".join(row["teacher_knowledge_conflicts"]),
        "rationale_generation_error": row.get("rationale_generation_error", ""),
    }
    for task in TASKS:
        flat[f"state_{task}"] = (row.get("teacher_states") or {}).get(task, "")
        flat[f"rationale_{task}"] = (row.get("structured_rationale") or {}).get(task, "")
    return flat


def write_csv(path, rows):
    flat = [flatten_for_csv(row) for row in rows]
    fields = list(flat[0]) if flat else ["sample_id", "audit_status"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def build_summary(audited, expected_total, manual_count):
    statuses = Counter(row["audit_status"] for row in audited)
    generated = (
        statuses.get("usable_valid", 0)
        + statuses.get("invalid_generated_package", 0)
        + statuses.get("generation_failed", 0)
        + statuses.get("invalid_generation_failure", 0)
    )
    defect_counts = Counter(
        (row.get("teacher_states") or {}).get("primary_defect", "unknown")
        for row in audited if row["audit_status"] == "usable_valid"
    )
    rule_counts = Counter(
        rule for row in audited for rule in row.get("retrieved_rule_ids", [])
    )
    expected_total = max(int(expected_total), len(audited), 1)
    return {
        "expected_total_records": expected_total,
        "written_records": len(audited),
        "pending_records": max(0, expected_total - len(audited)),
        "completion_rate": len(audited) / expected_total,
        "status_counts": dict(statuses),
        "generated_records": generated,
        "generated_content_pass_rate": statuses.get("usable_valid", 0) / max(1, generated),
        "conflict_mask_rate": statuses.get("masked_conflict", 0) / expected_total,
        "valid_defect_counts": dict(defect_counts),
        "retrieved_rule_counts": dict(rule_counts),
        "manual_review_records": manual_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manual-per-defect", type=int, default=8)
    parser.add_argument("--manual-conflicts", type=int, default=20)
    parser.add_argument("--expected-total", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = manifest_image_index(args.manifest)
    audited = []
    for record in read_jsonl(args.packages):
        if not record.get("image_path"):
            for key in (record.get("sample_id"), record.get("segment_id"), record.get("review_id")):
                if str(key or "") in images:
                    record["image_path"] = images[str(key)]
                    break
        audited.append(audit_record(record))

    manual = stratified_manual_sample(
        audited,
        per_defect=args.manual_per_defect,
        conflict_limit=args.manual_conflicts,
    )
    summary = build_summary(
        audited,
        expected_total=args.expected_total or len(audited),
        manual_count=len(manual),
    )

    with (out / "knowledge_package_audit_all.jsonl").open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "knowledge_package_manual_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in manual:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(out / "knowledge_package_audit_all.csv", audited)
    write_csv(out / "knowledge_package_manual_review.csv", manual)
    (out / "knowledge_package_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
