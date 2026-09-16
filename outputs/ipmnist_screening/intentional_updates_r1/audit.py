"""Recompute and validate the supervised Intentional Updates development result."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    IPMNISTConfig,
    ScreeningRunResult,
    intentional_updates_development_record,
    load_shard,
    merge_shards,
    validate_intentional_updates_development_record,
)

ROOT = Path(__file__).resolve().parent
ARMS = (
    "intentional_updates_ipmnist",
    "intentional_updates_no_diag",
    "intentional_updates_no_clip",
    "intentional_updates_head_only",
    "intentional_updates_off",
)
SEEDS = tuple(range(1_561_000, 1_561_005))
CONTROL = "intentional_updates_off"
CANDIDATE = "intentional_updates_ipmnist"
EXPECTED_CONFIG = {
    "n_tasks": 60,
    "task_length": 5000,
    "input_dim": 784,
    "hidden1": 300,
    "hidden2": 150,
    "n_classes": 10,
}


def _strict_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: invalid JSON constant {value}")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if type(value) is not dict:
        raise ValueError(f"{path}: expected one exact JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_sd_stderr(values: list[float]) -> dict[str, object]:
    if len(values) != len(SEEDS) or not all(math.isfinite(value) for value in values):
        raise ValueError("aggregate inputs must contain five finite values")
    sd = statistics.stdev(values)
    return {
        "values_by_seed": values,
        "mean": statistics.fmean(values),
        "sample_standard_deviation": sd,
        "standard_error": sd / math.sqrt(len(values)),
    }


def build_audit() -> dict[str, object]:
    paths = sorted((ROOT / "shards").glob("*.json"))
    expected_pairs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(paths) != len(expected_pairs):
        raise ValueError("campaign must contain exactly 25 shard files")

    summary_path = ROOT / "summary.json"
    summary = _strict_object(summary_path)
    recomputed = merge_shards(paths, control_name=CONTROL, slope_window=15)
    repository_root = ROOT.parents[2]
    for manifest_row in recomputed["shard_manifest"]:
        manifest_row["path"] = str(Path(manifest_row["path"]).relative_to(repository_root))
    for payload in (summary, recomputed):
        payload.pop("created_unix", None)
    if summary != recomputed:
        raise ValueError("persisted summary does not equal a fresh strict merge")

    rows: dict[tuple[str, int], dict[str, object]] = {}
    receipts: dict[tuple[str, int], dict[str, object]] = {}
    source_identities: set[str] = set()
    dataset_identities: set[str] = set()
    runtime_identities: set[str] = set()
    shard_manifest: list[dict[str, object]] = []
    for path in paths:
        shard = load_shard(path)
        arm = shard["config_name"]
        seed = shard["seed"]
        pair = (arm, seed)
        if pair not in expected_pairs or pair in rows:
            raise ValueError(f"unexpected or duplicate campaign pair: {pair}")
        if shard["config"] != EXPECTED_CONFIG:
            raise ValueError(f"{path}: protocol configuration drift")
        config = IPMNISTConfig(**shard["config"])
        result = ScreeningRunResult(
            config_name=arm,
            base_learner=shard["base_learner"],
            hyperparameters=shard["hyperparameters"],
            seed=seed,
            config=config,
            per_task_accuracy=np.asarray(shard["per_task_accuracy"], dtype=np.float64),
            per_task_loss=np.asarray(shard["per_task_loss"], dtype=np.float64),
            per_task_plasticity=np.asarray(
                shard["per_task_plasticity"], dtype=np.float64
            ),
            wall_clock_seconds=shard["wall_clock_seconds"],
            noise_mode=shard["noise_mode"],
            noise_pool_steps=shard["noise_pool_steps"],
        )
        receipt = validate_intentional_updates_development_record(
            intentional_updates_development_record(result)
        )
        source_text = json.dumps(shard["source_provenance"], sort_keys=True, separators=(",", ":"))
        dataset_text = json.dumps(
            shard["dataset_provenance"], sort_keys=True, separators=(",", ":")
        )
        runtime_text = json.dumps(shard["environment"], sort_keys=True, separators=(",", ":"))
        source_identities.add(hashlib.sha256(source_text.encode()).hexdigest())
        dataset_identities.add(hashlib.sha256(dataset_text.encode()).hexdigest())
        runtime_identities.add(hashlib.sha256(runtime_text.encode()).hexdigest())
        rows[pair] = shard
        receipts[pair] = receipt
        shard_manifest.append(
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "arm": arm, "seed": seed}
        )

    if set(rows) != expected_pairs:
        raise ValueError("campaign Cartesian product is incomplete")
    if len(source_identities) != 1 or len(dataset_identities) != 1 or len(runtime_identities) != 1:
        raise ValueError("campaign shards do not share one source/dataset/runtime identity")

    arm_results: dict[str, object] = {}
    expected_counters = {
        "observations": 300_000,
        "updates": 300_000,
        "backward_passes": 300_000,
        "model_queries": 600_000,
    }
    for arm in ARMS:
        accuracy = [
            statistics.fmean(rows[(arm, seed)]["per_task_accuracy"]) for seed in SEEDS
        ]
        loss = [statistics.fmean(rows[(arm, seed)]["per_task_loss"]) for seed in SEEDS]
        plasticity = [
            statistics.fmean(rows[(arm, seed)]["per_task_plasticity"]) for seed in SEEDS
        ]
        resource_rows = [receipts[(arm, seed)]["resources"] for seed in SEEDS]
        for resources in resource_rows:
            if any(resources[key] != value for key, value in expected_counters.items()):
                raise ValueError(f"{arm}: matched resource counter drift")
        persistent_bytes = {resources["persistent_numeric_bytes"] for resources in resource_rows}
        if len(persistent_bytes) != 1:
            raise ValueError(f"{arm}: persistent byte count changes by seed")
        arm_results[arm] = {
            "online_accuracy": _mean_sd_stderr(accuracy),
            "loss": _mean_sd_stderr(loss),
            "plasticity": _mean_sd_stderr(plasticity),
            "resources_per_seed": {
                **expected_counters,
                "persistent_numeric_bytes": persistent_bytes.pop(),
                "timing_telemetry_seconds": [
                    resources["timing_telemetry_seconds"] for resources in resource_rows
                ],
                "timing_is_selection_metric": False,
            },
        }

    candidate = arm_results[CANDIDATE]["online_accuracy"]["values_by_seed"]
    control = arm_results[CONTROL]["online_accuracy"]["values_by_seed"]
    deltas = [
        candidate_value - control_value
        for candidate_value, control_value in zip(candidate, control, strict=True)
    ]
    paired = _mean_sd_stderr(deltas)
    mean_delta = paired["mean"]
    all_positive = all(delta > 0.0 for delta in deltas)
    if mean_delta > 0.005 and all_positive:
        decision = "development_win"
    elif mean_delta <= 0.0:
        decision = "development_rejection"
    else:
        decision = "inconclusive"

    source = summary["source_provenance"]
    return {
        "schema": "asi.ipmnist.intentional_updates.supervised_audit.v1",
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "reference_dev_changed": False,
        },
        "plan": {
            "path": "PREREGISTRATION.md",
            "sha256": _sha256(ROOT / "PREREGISTRATION.md"),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "config": EXPECTED_CONFIG,
            "primary_candidate": CANDIDATE,
            "primary_control": CONTROL,
            "win_threshold_strictly_greater_than": 0.005,
            "all_paired_seeds_must_improve": True,
        },
        "execution": {
            "source_commit": source["git_commit"],
            "relevant_source_sha256": source["relevant_source_sha256"],
            "source_identity_sha256": next(iter(source_identities)),
            "dataset_identity_sha256": next(iter(dataset_identities)),
            "runtime_identity_sha256": next(iter(runtime_identities)),
            "summary_sha256": _sha256(summary_path),
            "shards": shard_manifest,
        },
        "results": arm_results,
        "primary_comparison": {
            "candidate": CANDIDATE,
            "control": CONTROL,
            "paired_online_accuracy_delta": paired,
            "all_paired_seeds_positive": all_positive,
            "decision": decision,
        },
        "audit_source_sha256": _sha256(Path(__file__)),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit.py OUTPUT")
    output = Path(sys.argv[1])
    payload = build_audit()
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(json.dumps(payload["primary_comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
