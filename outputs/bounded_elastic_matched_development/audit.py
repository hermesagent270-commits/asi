"""Independently audit the retained #1562 matched development result."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, cast

MEASURED_SOURCE = "8f6cea1705f774eab61272d3afd636265010c588"
PLAN_SHA256 = "4c7a991acf81f3846c00b73e55d8e80bf6e8b23486996a8523e6a6000e70e02e"
RESULT_DIGEST = "17681f37d707afa90276672ff59bb7420c896de4b5a43f43756176bd176fc613"
SEEDS = (51_562_001, 51_562_002, 51_562_003, 51_562_004, 51_562_005)
ARMS = (
    "bounded_structure_off",
    "bounded_growth",
    "bounded_elastic",
    "bounded_fixed_cbp",
)
CONFIG = {
    "n_tasks": 8,
    "task_length": 5000,
    "input_dim": 784,
    "hidden1": 300,
    "hidden2": 150,
    "n_classes": 10,
}
POLICY = {
    "completed_outcomes_retained_in_result": True,
    "development_only": True,
    "post_dispatch_failure_tombstone_retained": True,
    "post_dispatch_retry_prevention": True,
    "pre_dispatch_failure_receipts_retained": False,
    "reservation_precedes_execution_and_publication": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
}
STATIC_RESOURCES = {
    "bounded_structure_off": {
        "persistent_bytes": 1_130_144,
        "final_active_hidden1_units": 150,
        "peak_active_hidden1_units": 150,
        "final_active_parameter_bytes": 567_640,
        "structure_events": 0,
        "units_grown": 0,
        "units_pruned": 0,
    },
    "bounded_growth": {
        "persistent_bytes": 1_130_144,
        "final_active_hidden1_units": 158,
        "peak_active_hidden1_units": 158,
        "final_active_parameter_bytes": 597_560,
        "structure_events": 8,
        "units_grown": 8,
        "units_pruned": 0,
    },
    "bounded_elastic": {
        "persistent_bytes": 1_130_144,
        "final_active_hidden1_units": 150,
        "peak_active_hidden1_units": 150,
        "final_active_parameter_bytes": 567_640,
        "structure_events": 8,
        "units_grown": 8,
        "units_pruned": 8,
    },
    "bounded_fixed_cbp": {
        "persistent_bytes": 1_132_248,
        "final_active_hidden1_units": 300,
        "peak_active_hidden1_units": 300,
        "final_active_parameter_bytes": 1_128_640,
        "structure_events": 0,
        "units_grown": 0,
        "units_pruned": 0,
    },
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one exact object")
    return cast(dict[str, object], value)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _same(left: object, right: object) -> bool:
    return _canonical(left) == _canonical(right)


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be one finite float")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source(root: Path, source_identity: object) -> None:
    if type(source_identity) is not dict:
        raise ValueError("source identity must be an exact object")
    for relative, expected in cast(dict[object, object], source_identity).items():
        if type(relative) is not str or type(expected) is not str:
            raise ValueError("source identity entries must be strings")
        content = subprocess.check_output(
            ["git", "show", f"{MEASURED_SOURCE}:{relative}"], cwd=root
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"measured source mismatch: {relative}")


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _summary(values: list[float]) -> dict[str, object]:
    sample_sd = statistics.stdev(values)
    return {
        "values": values,
        "mean": _mean(values),
        "sample_sd": sample_sd,
        "standard_error": sample_sd / math.sqrt(len(values)),
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    report_path = here / "report.v1.json"
    plan_path = here / "plan.v1.json"
    audit_path = here / "audit.v1.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to replace {audit_path}")

    report = _load(report_path)
    plan = _load(plan_path)
    if hashlib.sha256(_canonical(plan)).hexdigest() != PLAN_SHA256:
        raise ValueError("plan digest differs from the preregistered digest")
    if report.get("schema") != "asi.bounded-elastic-ipmnist.matched-development.v1":
        raise ValueError("result schema drifted")
    if report.get("status") != "complete":
        raise ValueError("result is not complete")
    if report.get("result_sha256") != RESULT_DIGEST:
        raise ValueError("registered result digest drifted")
    unsigned = dict(report)
    unsigned.pop("result_sha256")
    if hashlib.sha256(_canonical(unsigned)).hexdigest() != RESULT_DIGEST:
        raise ValueError("canonical result digest does not verify")
    if not _same(report.get("development_seeds"), list(SEEDS)):
        raise ValueError("development roster drifted")
    if not _same(report.get("arms"), list(ARMS)):
        raise ValueError("arm roster drifted")
    if not _same(report.get("config"), CONFIG):
        raise ValueError("campaign config drifted")
    if not _same(report.get("policy"), POLICY):
        raise ValueError("nonpromoting policy drifted")

    identity = report.get("identity")
    if type(identity) is not dict:
        raise ValueError("result identity must be an exact object")
    typed_identity = cast(dict[str, object], identity)
    if typed_identity.get("dataset_sha256") != (
        "c1c4bae4e0981268b32bb802b193a9107b665590e9fc80cdae4dd6ac4e4b78ae"
    ):
        raise ValueError("dataset identity drifted")
    if typed_identity.get("consistency_not_attestation") is not True:
        raise ValueError("consistency limitation is missing")
    _verify_source(root, typed_identity.get("source_sha256"))
    if not _same(plan.get("source_identity"), typed_identity.get("source_sha256")):
        raise ValueError("plan and result source identities differ")

    rows = report.get("rows")
    if type(rows) is not list:
        raise ValueError("rows must be an exact list")
    typed_rows = cast(list[dict[str, Any]], rows)
    expected_roster = [(seed, arm) for seed in SEEDS for arm in ARMS]
    actual_roster = [(row.get("seed"), row.get("arm")) for row in typed_rows]
    if actual_roster != expected_roster:
        raise ValueError("row roster or order drifted")

    metrics_by_arm: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    timing_by_arm: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for row in typed_rows:
        seed = cast(int, row["seed"])
        arm = cast(str, row["arm"])
        result = cast(dict[str, Any], row.get("result"))
        if result.get("seed") != seed or result.get("arm") != arm:
            raise ValueError("row and nested result identity differ")
        if result.get("observations") != 40_000 or result.get("updates") != 40_000:
            raise ValueError("row step budget drifted")
        if result.get("outcome") != "inconclusive" or result.get("outcome_retained") is not True:
            raise ValueError("row outcome retention drifted")
        if result.get("development_only") is not True:
            raise ValueError("row development policy drifted")
        if result.get("scientific_promotion_allowed") is not False:
            raise ValueError("row promotion policy drifted")
        resources = cast(dict[str, Any], result.get("resources"))
        common_resources = {
            "data_steps": 40_000,
            "environment_steps": 0,
            "model_queries": 80_000,
            "peak_parameter_bytes": 1_128_640,
            "final_active_parameter_bytes_budget": 1_128_640,
            "peak_persistent_bytes_budget": 1_132_248,
            "timing_is_telemetry_only": True,
        }
        for key, expected in {**common_resources, **STATIC_RESOURCES[arm]}.items():
            if resources.get(key) != expected:
                raise ValueError(f"resource drift for {arm}.{key}")
        timing_by_arm[arm].append(_finite_float(resources.get("timing_seconds"), "timing"))
        metrics = cast(dict[str, object], result.get("metrics"))
        checked_metrics = {
            key: _finite_float(metrics.get(key), f"{arm}.{key}")
            for key in ("mean_online_accuracy", "mean_loss", "mean_plasticity")
        }
        if not 0.0 <= checked_metrics["mean_online_accuracy"] <= 1.0:
            raise ValueError("online accuracy is outside [0,1]")
        if not 0.0 <= checked_metrics["mean_plasticity"] <= 1.0:
            raise ValueError("plasticity is outside [0,1]")
        metrics_by_arm[arm].append(checked_metrics)
        execution = cast(dict[str, object], row.get("execution_identity"))
        if execution.get("prng_implementation") != "threefry2x32":
            raise ValueError("row PRNG identity drifted")
        for key in ("schedule_sha256", "initial_parameters_sha256"):
            value = execution.get(key)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"row {key} is invalid")

    for offset in range(0, len(typed_rows), len(ARMS)):
        seed_rows = typed_rows[offset : offset + len(ARMS)]
        for key in ("schedule_sha256", "initial_parameters_sha256"):
            if len({row["execution_identity"][key] for row in seed_rows}) != 1:
                raise ValueError(f"matched seed rows differ in {key}")

    recomputed_arms: dict[str, object] = {}
    for arm in ARMS:
        recomputed_arms[arm] = {
            "mean_accuracy": _mean(
                [metric["mean_online_accuracy"] for metric in metrics_by_arm[arm]]
            ),
            "mean_loss": _mean([metric["mean_loss"] for metric in metrics_by_arm[arm]]),
            "mean_plasticity": _mean(
                [metric["mean_plasticity"] for metric in metrics_by_arm[arm]]
            ),
        }

    comparisons: list[dict[str, object]] = []
    comparison_stats: dict[str, object] = {}
    baseline = [
        metric["mean_online_accuracy"] for metric in metrics_by_arm["bounded_fixed_cbp"]
    ]
    for candidate in ("bounded_growth", "bounded_elastic"):
        deltas = [
            metric["mean_online_accuracy"] - control
            for metric, control in zip(metrics_by_arm[candidate], baseline, strict=True)
        ]
        outcome = (
            "supported"
            if all(delta > 0.0 for delta in deltas)
            else "rejected"
            if all(delta <= 0.0 for delta in deltas)
            else "inconclusive"
        )
        comparisons.append(
            {
                "candidate": candidate,
                "baseline": "bounded_fixed_cbp",
                "paired_accuracy_deltas": deltas,
                "mean_accuracy_delta": _mean(deltas),
                "outcome": outcome,
            }
        )
        comparison_stats[candidate] = _summary(deltas)
    campaign_outcome = (
        "supported"
        if any(item["outcome"] == "supported" for item in comparisons)
        else "rejected"
        if all(item["outcome"] == "rejected" for item in comparisons)
        else "inconclusive"
    )
    recomputed_aggregate = {
        "row_count": 20,
        "arms": recomputed_arms,
        "primary_comparisons": comparisons,
        "outcome": campaign_outcome,
    }
    if not _same(report.get("aggregate"), recomputed_aggregate):
        raise ValueError("stored aggregate differs from independent recomputation")

    audit = {
        "schema": "asi.bounded-elastic-ipmnist.independent-audit.v1",
        "measured_source_commit": MEASURED_SOURCE,
        "plan_sha256": PLAN_SHA256,
        "result_digest": RESULT_DIGEST,
        "report_file_sha256": _sha256(report_path),
        "plan_file_sha256": _sha256(plan_path),
        "audit_script_sha256": _sha256(Path(__file__)),
        "row_count": 20,
        "source_files_verified_from_measured_commit": len(
            cast(dict[str, object], typed_identity["source_sha256"])
        ),
        "dataset_sha256": typed_identity["dataset_sha256"],
        "aggregate": recomputed_aggregate,
        "arm_accuracy_statistics": {
            arm: _summary(
                [metric["mean_online_accuracy"] for metric in metrics_by_arm[arm]]
            )
            for arm in ARMS
        },
        "comparison_statistics": comparison_stats,
        "initial_execution_timing_seconds": {
            arm: {"per_seed": timing_by_arm[arm], "total": math.fsum(timing_by_arm[arm])}
            for arm in ARMS
        },
        "transaction_resources": plan["transaction_resource_accounting"],
        "decision": campaign_outcome,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "consistency_hashes_are_not_execution_attestation": True,
        "checks": [
            "strict JSON and canonical result digest",
            "preregistered plan digest",
            "exact Cartesian row roster and ordering",
            "measured source bytes recovered from git",
            "plan/result source identity agreement",
            "dataset and explicit Threefry execution identities",
            "matched schedules and initial parameters within each seed",
            "step, query, memory, size, and structural-event resources",
            "arm aggregates, paired deltas, sign-rule decisions, and campaign outcome",
            "permanently nonpromoting retention policy",
        ],
    }
    encoded = _canonical(audit) + b"\n"
    descriptor = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("audit write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(audit, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
