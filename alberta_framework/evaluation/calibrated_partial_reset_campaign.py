"""Hard-disabled CPR campaign through the authoritative IPMNIST runner.

Adapted from the closed #2125 campaign contract, without its parallel learner.
Exposed development roots are never eligible for scientific promotion.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    CPR_OFFICIAL_CODE_REVISION as OFFICIAL_CODE_REVISION,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    CPR_PAPER_REVISION as PAPER_REVISION,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    _partial_reset_peak_numeric_bytes,
    _screening_dataset_provenance,
    _screening_source_provenance,
    partial_reset_development_record,
    run_screening_config,
    screening_spec,
    validate_partial_reset_development_record,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)

SCHEMA: Final = "asi.calibrated-partial-reset.matched-development.v2"
ARMS: Final = ("cpr_off", "cpr_ipmnist", "cpr_utility_free", "cpr_l2_init", "cpr_hard_reset")
CAMPAIGN_SEEDS: Final = (1_563_066_001, 1_563_066_002, 1_563_066_003, 1_563_066_004, 1_563_066_005)
TEST_ONLY_SEEDS: Final = (301, 302, 303, 304, 305)
CAMPAIGN_CONFIG: Final = IPMNISTConfig(n_tasks=60, task_length=5000)
_REVIEWED_EXECUTION_TRANSITION: Final = False
_EXECUTION_AUTHORIZED: Final = False
_EXECUTION_CAPABILITY: Final = object()
_TEST_EXECUTION_CAPABILITY: Final = object()
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_DEPTH: Final = 16
_MAX_STRING_BYTES: Final = 16_384
_MAX_TOTAL_UTF8_BYTES: Final = 8 * 1024 * 1024
_MIN_INTEGER: Final = -(2**63)
_MAX_INTEGER: Final = 2**63 - 1
_MAX_REPORT_BYTES: Final = 64 * 1024 * 1024
_MAX_NUMERIC_BYTES: Final = 256 * 1024 * 1024
_MAX_RUNTIME_DEVICES: Final = 64
_ROOT: Final = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final = _ROOT / "outputs/calibrated_partial_reset_matched/development.v2/report.json"
_PRODUCTION_NAMESPACE: Final = OUTPUT_PATH.parent
_PRIMARY_RULE: Final = (
    "advance iff mean_delta > 0.005 and all five seed deltas > 0; "
    "reject iff mean_delta <= 0; otherwise inconclusive"
)
FROZEN_PLAN_SHA256: Final = "c3f832cc25d770d33e9027d3c843667b15255ebba1485c34b2098098fce43cff"


def _require_authorized() -> None:
    if _REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True:
        raise RuntimeError("CPR campaign execution is not authorized")
    frozen_plan()


def _check_request(config: IPMNISTConfig, seeds: tuple[int, ...]) -> bool:
    if type(seeds) is not tuple or len(seeds) != 5 or any(type(seed) is not int for seed in seeds):
        raise ValueError("CPR seeds must be an exact five-integer tuple")
    production = seeds == CAMPAIGN_SEEDS
    if not production and seeds != TEST_ONLY_SEEDS:
        raise ValueError("CPR seed roster differs from the campaign or test-only contract")
    if type(config) is not IPMNISTConfig:
        raise ValueError("CPR config must be an exact IPMNISTConfig")
    if production:
        if not _exact_json_equal(config.to_config(), CAMPAIGN_CONFIG.to_config()):
            raise ValueError("CPR campaign config differs from the frozen plan")
    elif config.n_steps > 512 or config.input_dim > 16 or config.parameter_count > 10_000:
        raise ValueError("CPR test-only execution exceeds its small contract bound")
    return production


def _require_test_namespace(destination: Path) -> None:
    if type(destination) is not type(Path()) or destination.absolute().is_relative_to(
        _PRODUCTION_NAMESPACE
    ):
        raise RuntimeError("CPR test-only namespace must be outside the production namespace")


def _source_identity() -> dict[str, str]:
    paths = sorted(
        path.relative_to(_ROOT).as_posix() for path in (_ROOT / "alberta_framework").rglob("*.py")
    )
    paths.extend(("pyproject.toml", "uv.lock"))
    return {path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest() for path in paths}


def _runtime_identity() -> dict[str, object]:
    devices = tuple(jax.devices())
    if len(devices) == 0 or len(devices) > _MAX_RUNTIME_DEVICES:
        raise RuntimeError("CPR runtime device inventory is out of bounds")
    default_prng = str(jax.config.jax_default_prng_impl)
    random_seed_offset = int(jax.config.jax_random_seed_offset)
    if default_prng != "threefry2x32" or random_seed_offset != 0:
        raise RuntimeError("CPR execution requires unoffset Threefry RNG roots")
    environment = (
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_ENABLE_X64",
        "JAX_NUM_CPU_DEVICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_RANDOM_SEED_OFFSET",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "XLA_FLAGS",
    )
    distributions = tuple(importlib.metadata.distributions())
    if len(distributions) > 1024:
        raise RuntimeError("CPR dependency inventory exceeds its bound")
    packages: dict[str, str] = {}
    for distribution in distributions:
        name = distribution.metadata.get("Name")
        version = distribution.version
        if type(name) is not str or type(version) is not str:
            raise RuntimeError("CPR dependency inventory lacks exact name/version metadata")
        normalized = name.lower().replace("_", "-")
        if normalized in packages:
            raise RuntimeError("CPR dependency inventory contains duplicate distributions")
        packages[normalized] = version
    config_values = {
        name: value.value if isinstance(value, Enum) else value
        for name, value in jax.config.values.items()
    }
    _bounded_json(config_values)
    return {
        "schema": "asi.calibrated-partial-reset.runtime.v1",
        "python": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": dict(sorted(packages.items())),
        "jax": {
            "backend": jax.default_backend(),
            "devices": [
                {
                    "id": int(device.id),
                    "platform": str(device.platform),
                    "device_kind": str(device.device_kind),
                    "process_index": int(device.process_index),
                }
                for device in devices
            ],
            "config": {
                "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
                "jax_default_prng_impl": default_prng,
                "jax_disable_jit": bool(jax.config.jax_disable_jit),
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion.value),
                "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
                "jax_random_seed_offset": random_seed_offset,
                "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
            },
            "all_config_values": config_values,
        },
        "environment": {name: os.environ.get(name) for name in environment},
    }


def _resource_envelope(config: IPMNISTConfig, rows: int) -> dict[str, int]:
    dataset = rows * (config.input_dim * 4 + 4)
    schedule = config.n_tasks * (config.task_length + config.input_dim) * 4
    persistent = _partial_reset_peak_numeric_bytes(config)
    return {
        "dataset_bytes": dataset,
        "schedule_bytes": schedule,
        "peak_persistent_numeric_bytes": persistent,
        "combined_numeric_bytes": dataset + schedule + persistent,
        "combined_numeric_bytes_limit": _MAX_NUMERIC_BYTES,
    }


def _transaction_resources(config: IPMNISTConfig, seeds: tuple[int, ...]) -> dict[str, int]:
    rows = len(seeds) * len(ARMS)
    dispatches = 2 * rows
    steps = config.n_tasks * config.task_length
    return {
        "campaign_rows": rows,
        "initial_runner_dispatches": rows,
        "strict_reexecution_dispatches": rows,
        "total_runner_dispatches": dispatches,
        "total_observations": dispatches * steps,
        "total_updates": dispatches * steps,
        "total_data_steps": dispatches * steps,
        "total_environment_steps": 0,
        "total_model_queries": 2 * dispatches * steps,
    }


def _plan_payload() -> dict[str, object]:
    return {
        "schema": "asi.calibrated-partial-reset.matched-plan.v2",
        "paper_revision": PAPER_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "upstream_alignment": {
            "source_path": "continual_learning/optim/cpr.py",
            "source_git_blob_sha1": "b6bd13bbd0a519b122670c19729df8021d815d0d",
            "reset_timing": "positive pre-update clock; frequency F first acts at update F+1",
            "reset_timing_source_lines": [277, 280],
            "utility_recentering": "ones after each periodic reset",
            "utility_recentering_source_lines": [256, 263],
        },
        "adaptation": {
            "workload": "ASI online IPMNIST instead of the paper RL suites",
            "optimizer": "matched float32 SGD with online EMA input normalization",
            "initialization": "retained seed initialization, not fresh reset draws",
            "reset_axis": "all parameters, including biases and output weights",
            "utility": "per-parameter absolute-gradient EMA normalized by tensor mean",
        },
        "seeds": list(CAMPAIGN_SEEDS),
        "test_only_seeds": list(TEST_ONLY_SEEDS),
        "arms": list(ARMS),
        "config": CAMPAIGN_CONFIG.to_config(),
        "dataset": {
            "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
            "source": {
                "provider": "openml",
                "name": "mnist_784",
                "version": 1,
                "row_start": 0,
                "row_stop_exclusive": 60_000,
            },
            "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
            "x": {
                "dtype": "<f4",
                "shape": [60_000, 784],
                "sha256": "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313",
            },
            "y": {
                "dtype": "<i4",
                "shape": [60_000],
                "sha256": "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a",
            },
        },
        "source_identity_policy": (
            "all alberta_framework Python files plus pyproject.toml and uv.lock"
        ),
        "arm_hyperparameters": {arm: dict(screening_spec(arm).hyperparameters) for arm in ARMS},
        "runtime_identity_policy": "exact dependencies/JAX devices/config/environment at execution",
        "resources": _resource_envelope(CAMPAIGN_CONFIG, 60_000),
        "resource_scope": (
            "one caller-owned C-contiguous host dataset, one materialized schedule, and "
            "parameters plus retained learner state; backend/compiler copies, gradients, "
            "and transient execution buffers are excluded from the static byte envelope"
        ),
        "transaction_resources": _transaction_resources(CAMPAIGN_CONFIG, CAMPAIGN_SEEDS),
        "matched_axes": [
            "seed",
            "initial_parameters",
            "example_schedule",
            "observations",
            "updates",
            "allowed_boundary_information",
            "allowed_task_information",
        ],
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "primary_paired_question": {
            "candidate": "cpr_ipmnist",
            "control": "cpr_off",
            "metric": "mean_online_accuracy",
            "direction": "higher_is_better",
            "decision_rule": _PRIMARY_RULE,
            "advance_outcome": "advance_for_nonpromoting_followup",
            "reject_outcome": "do_not_advance",
            "inconclusive_outcome": "inconclusive",
            "ablation_policy": "utility-free, L2-init, and hard-reset arms are descriptive only",
        },
        "reviewed_execution_transition": _REVIEWED_EXECUTION_TRANSITION,
        "execution_authorized": _EXECUTION_AUTHORIZED,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
        "first_dispatch_policy": (
            "the complete roster is consumed before the first learner call; "
            "reservations and tombstones prohibit retry"
        ),
        "failure_scope": (
            "ordinary exceptions retain a tombstone when the filesystem permits; "
            "process-death recovery and durable result publication are not guaranteed"
        ),
        "output_path": "outputs/calibrated_partial_reset_matched/development.v2/report.json",
    }


def frozen_plan() -> dict[str, object]:
    """Return the literal-hash-bound, permanently nonpromoting development plan."""
    plan = _plan_payload()
    if hashlib.sha256(_canonical(plan)).hexdigest() != FROZEN_PLAN_SHA256:
        raise RuntimeError("CPR frozen plan digest drifted; a reviewed revision is required")
    return plan


def _bounded_json(value: object) -> object:
    nodes = 0
    utf8_bytes = 0
    seen: set[int] = set()

    def visit(item: object, depth: int) -> object:
        nonlocal nodes, utf8_bytes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("CPR result exceeds its JSON structure bound")
        actual_type = type(item)
        if item is None or actual_type is bool:
            return item
        if actual_type is int:
            if cast(int, item) < _MIN_INTEGER or cast(int, item) > _MAX_INTEGER:
                raise ValueError("CPR result contains an out-of-bounds integer")
            return item
        if actual_type is float:
            if not math.isfinite(cast(float, item)):
                raise ValueError("CPR result contains a non-finite float")
            return item
        if actual_type is str:
            size = len(cast(str, item).encode("utf-8"))
            utf8_bytes += size
            if size > _MAX_STRING_BYTES or utf8_bytes > _MAX_TOTAL_UTF8_BYTES:
                raise ValueError("CPR result contains an oversized string")
            return item
        if actual_type is not dict and actual_type is not list:
            raise ValueError("CPR result must contain only exact JSON values")
        identity = id(item)
        if identity in seen:
            raise ValueError("CPR result contains an aliased or cyclic container")
        seen.add(identity)
        if actual_type is list:
            sequence = cast(list[object], item)
            if len(sequence) > 4096:
                raise ValueError("CPR result list exceeds its item bound")
            return [visit(child, depth + 1) for child in sequence]
        mapping = cast(dict[object, object], item)
        if len(mapping) > 4096 or any(type(key) is not str for key in mapping):
            raise ValueError("CPR result object exceeds its field bound")
        result: dict[str, object] = {}
        for key, child in mapping.items():
            checked_key = cast(str, key)
            key_size = len(checked_key.encode("utf-8"))
            utf8_bytes += key_size
            if key_size > _MAX_STRING_BYTES or utf8_bytes > _MAX_TOTAL_UTF8_BYTES:
                raise ValueError("CPR result contains oversized aggregate UTF-8")
            result[checked_key] = visit(child, depth + 1)
        return result

    return visit(value, 0)


def _validated_arrays(x: object, y: object, config: IPMNISTConfig) -> tuple[np.ndarray, np.ndarray]:
    if type(x) is not np.ndarray or x.dtype != np.dtype(np.float32):
        raise ValueError("inputs must be an exact float32 NumPy array")
    if type(y) is not np.ndarray or y.dtype != np.dtype(np.int32):
        raise ValueError("labels must be an exact int32 NumPy array")
    if x.ndim != 2 or y.ndim != 1 or x.shape != (y.shape[0], config.input_dim):
        raise ValueError("dataset shape differs from the CPR config")
    if x.shape[0] < config.task_length:
        raise ValueError("dataset has too few rows for the CPR schedule")
    if not x.flags.c_contiguous or not y.flags.c_contiguous:
        raise ValueError("dataset arrays must be C-contiguous to avoid an unaccounted copy")
    envelope = _resource_envelope(config, x.shape[0])
    if envelope["combined_numeric_bytes"] > _MAX_NUMERIC_BYTES:
        raise ValueError("CPR campaign exceeds its combined numeric allocation bound")
    if not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= config.n_classes):
        raise ValueError("dataset numeric domain differs from the CPR contract")
    return x, y


def _tree_digest(value: object) -> str:
    digest = hashlib.sha256(b"asi-cpr-tree-v1\0")
    leaves, structure = jax.tree.flatten(value)
    digest.update(str(structure).encode("ascii"))
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _dataset_identity(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    if x.shape == (60_000, 784) and y.shape == (60_000,):
        return _screening_dataset_provenance(x, y)
    return {
        "schema": "asi.calibrated-partial-reset.test-dataset.v1",
        "x": {
            "dtype": x.dtype.str,
            "shape": list(x.shape),
            "sha256": hashlib.sha256(x.tobytes(order="C")).hexdigest(),
        },
        "y": {
            "dtype": y.dtype.str,
            "shape": list(y.shape),
            "sha256": hashlib.sha256(y.tobytes(order="C")).hexdigest(),
        },
    }


def _execution_identity(seed: int, config: IPMNISTConfig, rows: int) -> dict[str, str]:
    root = jr.key(np.uint32(seed), impl="threefry2x32")
    init_key, schedule_key, noise_key = jr.split(root, 3)
    schedule = build_schedule(schedule_key, config, rows)
    params = init_mlp_params(init_key, config)
    spec = screening_spec(ARMS[0])
    init_fn, _ = spec.factory(spec.hyperparameters)
    return {
        "initial_parameters_sha256": _tree_digest(params),
        "initial_learner_state_sha256": _tree_digest(init_fn(params)),
        "schedule_sha256": _tree_digest(schedule),
        "initial_learner_rng_sha256": _tree_digest(jr.key_data(noise_key)),
        "prng_implementation": "threefry2x32",
    }


def _record(result: ScreeningRunResult) -> dict[str, object]:
    return cast(dict[str, object], partial_reset_development_record(result))


def _mean_accuracy(row: dict[str, object]) -> float:
    result = cast(dict[str, object], row["result"])
    metrics = cast(dict[str, object], result["metrics"])
    accuracy = cast(list[float], metrics["per_task_accuracy"])
    return math.fsum(accuracy) / len(accuracy)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    means: dict[str, float] = {}
    for arm in ARMS:
        values = [_mean_accuracy(row) for row in rows if row["arm"] == arm]
        means[arm] = math.fsum(values) / len(values)
    by_pair = {(cast(int, row["seed"]), cast(str, row["arm"])): _mean_accuracy(row) for row in rows}
    paired_deltas = [
        {
            "seed": seed,
            "utility_minus_off": by_pair[(seed, "cpr_ipmnist")] - by_pair[(seed, "cpr_off")],
        }
        for seed in sorted({seed for seed, _arm in by_pair})
    ]
    delta_values = [item["utility_minus_off"] for item in paired_deltas]
    mean_delta = math.fsum(delta_values) / len(delta_values)
    positive_count = sum(delta > 0.0 for delta in delta_values)
    outcome = (
        "advance_for_nonpromoting_followup"
        if mean_delta > 0.005 and positive_count == 5
        else "do_not_advance"
        if mean_delta <= 0.0
        else "inconclusive"
    )
    return {
        "mean_online_accuracy": means,
        "row_count": len(rows),
        "primary_paired_question": {
            "candidate": "cpr_ipmnist",
            "control": "cpr_off",
            "metric": "mean_online_accuracy",
            "direction": "higher_is_better",
            "paired_deltas": paired_deltas,
            "mean_delta": mean_delta,
            "positive_seed_count": positive_count,
            "decision_rule": _PRIMARY_RULE,
            "sample_standard_deviation": math.sqrt(
                math.fsum((delta - mean_delta) ** 2 for delta in delta_values)
                / (len(delta_values) - 1)
            ),
            "standard_error": math.sqrt(
                math.fsum((delta - mean_delta) ** 2 for delta in delta_values)
                / (len(delta_values) * (len(delta_values) - 1))
            ),
            "outcome": outcome,
        },
        "ablation_policy": "utility-free, L2-init, and hard-reset arms are descriptive only",
        "outcome": outcome,
    }


def _run(
    x: object,
    y: object,
    *,
    config: IPMNISTConfig,
    seeds: tuple[int, ...],
    capability: object,
    on_dispatch: Callable[[], None] | None = None,
) -> dict[str, object]:
    expected = (
        CAMPAIGN_SEEDS
        if capability is _EXECUTION_CAPABILITY
        else TEST_ONLY_SEEDS
        if capability is _TEST_EXECUTION_CAPABILITY
        else None
    )
    if (
        expected is None
        or type(seeds) is not tuple
        or len(seeds) != len(expected)
        or any(type(seed) is not int for seed in seeds)
        or any(seed != expected[index] for index, seed in enumerate(seeds))
    ):
        raise RuntimeError("CPR private execution capability is invalid")
    if capability is _EXECUTION_CAPABILITY:
        _require_authorized()
    production = _check_request(config, seeds)
    checked_x, checked_y = _validated_arrays(x, y, config)
    source = _source_identity()
    runtime = _runtime_identity()
    dataset = _dataset_identity(checked_x, checked_y)
    execution_source = _screening_source_provenance(_ROOT) if production else None
    _bounded_json(source)
    _bounded_json(runtime)
    _bounded_json(frozen_plan())
    if capability is _EXECUTION_CAPABILITY:
        if (
            config != CAMPAIGN_CONFIG
            or _screening_dataset_provenance(checked_x, checked_y) != frozen_plan()["dataset"]
        ):
            raise ValueError("CPR campaign inputs differ from the frozen plan")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        identity = _execution_identity(seed, config, checked_x.shape[0])
        for arm in ARMS:
            if not rows and on_dispatch is not None:
                on_dispatch()
            result = run_screening_config(checked_x, checked_y, screening_spec(arm), seed, config)
            if (
                _source_identity() != source
                or _runtime_identity() != runtime
                or _dataset_identity(checked_x, checked_y) != dataset
            ):
                raise RuntimeError("CPR source, runtime, or dataset changed during execution")
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "execution_identity": dict(identity),
                    "result": _record(result),
                }
            )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "plan": frozen_plan(),
        "identity": {
            "source": source,
            "runtime": runtime,
            "dataset": dataset,
            "execution_source": execution_source,
        },
        "rows": rows,
        "aggregate": _aggregate(rows),
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
        },
    }
    report["sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    validate_report(report, checked_x, checked_y, config=config, seeds=seeds, reexecute=False)
    return report


def _canonical(value: object) -> bytes:
    bounded = _bounded_json(value)
    encoded = json.dumps(bounded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "ascii"
    )
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("CPR canonical JSON exceeds its byte bound")
    return encoded


def _exact_json_equal(left: object, right: object) -> bool:
    return _canonical(left) == _canonical(right)


def _exact_object(value: object, fields: tuple[str, ...], label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"CPR {label} must be an exact object")
    result = cast(dict[str, object], value)
    if len(result) != len(fields) or any(field not in result for field in fields):
        raise ValueError(f"CPR {label} fields drifted")
    return result


def _validate_record(value: object, config: IPMNISTConfig, seed: int, arm: str) -> None:
    record = validate_partial_reset_development_record(value)
    if not _exact_json_equal(value, record):
        raise ValueError("CPR exact nested receipt fields drifted")
    if (
        not _exact_json_equal(record["config"], config.to_config())
        or record["seed"] != seed
        or record["arm"] != arm
    ):
        raise ValueError("CPR result row identity drifted")
    if not _exact_json_equal(record["hyperparameters"], dict(screening_spec(arm).hyperparameters)):
        raise ValueError("CPR result hyperparameters drifted")


def validate_report(
    value: object,
    x: object,
    y: object,
    *,
    config: IPMNISTConfig,
    seeds: tuple[int, ...],
    reexecute: bool,
) -> None:
    if type(reexecute) is not bool:
        raise ValueError("CPR reexecution selector must be an exact bool")
    if (
        reexecute
        and type(seeds) is tuple
        and len(seeds) == 5
        and all(type(seed) is int for seed in seeds)
        and seeds == CAMPAIGN_SEEDS
    ):
        _require_authorized()
    production = _check_request(config, seeds)
    report = cast(dict[str, object], _bounded_json(value))
    report = _exact_object(
        report, ("schema", "plan", "identity", "rows", "aggregate", "policy", "sha256"), "report"
    )
    if report["schema"] != SCHEMA:
        raise ValueError("CPR report fields or schema drifted")
    if not _exact_json_equal(report["plan"], frozen_plan()):
        raise ValueError("CPR report plan drifted")
    checked_x, checked_y = _validated_arrays(x, y, config)
    if production and not _exact_json_equal(
        _dataset_identity(checked_x, checked_y), frozen_plan()["dataset"]
    ):
        raise ValueError("CPR campaign dataset differs from the frozen plan")
    identity = _exact_object(
        report["identity"], ("source", "runtime", "dataset", "execution_source"), "identity"
    )
    if not _exact_json_equal(
        identity,
        {
            "source": _source_identity(),
            "runtime": _runtime_identity(),
            "dataset": _dataset_identity(checked_x, checked_y),
            "execution_source": _screening_source_provenance(_ROOT) if production else None,
        },
    ):
        raise ValueError("CPR report identity drifted")
    if type(report["rows"]) is not list:
        raise ValueError("CPR rows must be an exact list")
    rows = cast(list[dict[str, object]], report["rows"])
    roster = [(seed, arm) for seed in seeds for arm in ARMS]
    if len(rows) != len(roster):
        raise ValueError("CPR report roster drifted")
    expected_identity = {
        seed: _execution_identity(seed, config, checked_x.shape[0]) for seed in seeds
    }
    source = _source_identity()
    runtime = _runtime_identity()
    dataset = _dataset_identity(checked_x, checked_y)
    for index, raw_row in enumerate(rows):
        row = _exact_object(raw_row, ("seed", "arm", "execution_identity", "result"), "row")
        seed, arm = roster[index]
        if (
            type(row["seed"]) is not int
            or row["seed"] != seed
            or type(row["arm"]) is not str
            or row["arm"] != arm
        ):
            raise ValueError("CPR report roster drifted")
        if not _exact_json_equal(row["execution_identity"], expected_identity[seed]):
            raise ValueError("CPR row identity drifted")
        _validate_record(row["result"], config, seed, arm)
    _exact_object(
        report["aggregate"],
        (
            "mean_online_accuracy",
            "row_count",
            "primary_paired_question",
            "ablation_policy",
            "outcome",
        ),
        "aggregate",
    )
    if not _exact_json_equal(report["aggregate"], _aggregate(rows)):
        raise ValueError("CPR aggregate arithmetic drifted")
    policy = _exact_object(
        report["policy"],
        ("development_only", "scientific_promotion_allowed", "negative_outcomes_retained"),
        "policy",
    )
    if not _exact_json_equal(
        policy,
        {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
        },
    ):
        raise ValueError("CPR result policy drifted")
    unsigned = dict(report)
    claimed = unsigned.pop("sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("CPR report digest drifted")
    if reexecute:
        for row in rows:
            seed, arm = cast(int, row["seed"]), cast(str, row["arm"])
            replay = _record(
                run_screening_config(checked_x, checked_y, screening_spec(arm), seed, config)
            )
            replay_resources = cast(dict[str, object], replay["resources"])
            claimed_resources = cast(
                dict[str, object], cast(dict[str, object], row["result"])["resources"]
            )
            replay_resources["timing_telemetry_seconds"] = claimed_resources[
                "timing_telemetry_seconds"
            ]
            if not _exact_json_equal(replay, row["result"]):
                raise ValueError("CPR row differs from strict current reexecution")
            if (
                _source_identity() != source
                or _runtime_identity() != runtime
                or _dataset_identity(checked_x, checked_y) != dataset
            ):
                raise RuntimeError("CPR identity changed during strict reexecution")


Reservation = tuple[int, str, str, int, int, int]


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0 or written > len(view):
            raise OSError("CPR marker write made no progress")
        view = view[written:]


def _require_live_parent(reservation: Reservation) -> None:
    directory = reservation[0]
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    visible = os.open(os.path.sep, flags)
    try:
        for component in OUTPUT_PATH.parent.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=visible)
            os.close(visible)
            visible = next_descriptor
        held, current = os.fstat(directory), os.fstat(visible)
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("CPR registered output parent identity changed")
    except OSError as error:
        raise RuntimeError("CPR registered output parent is unavailable") from error
    finally:
        os.close(visible)


def _mark_dispatched(reservation: Reservation) -> None:
    _owned(reservation)
    _require_live_parent(reservation)
    descriptor = reservation[3]
    os.lseek(descriptor, 0, os.SEEK_SET)
    data = b"asi-cpr-dispatched-consumed-v2\n"
    _write_all(descriptor, data)
    os.ftruncate(descriptor, len(data))
    os.fsync(descriptor)
    os.fsync(reservation[0])
    _require_live_parent(reservation)


def _open_parent(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in absolute.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("CPR output contains an unsafe path segment")
            try:
                os.mkdir(component, 0o755, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _reserve(path: Path) -> Reservation:
    if type(path) is not type(Path()) or path.absolute() != OUTPUT_PATH.absolute():
        raise ValueError("CPR output must be the exact frozen NEW path")
    directory = _open_parent(path)
    marker_name = f".{path.name}.reservation"
    marker_fd = -1
    try:
        probe = os.open(".", os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC, 0o600, dir_fd=directory)
        os.close(probe)
        try:
            os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("CPR output already exists")
        marker_fd = os.open(
            marker_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=directory,
        )
        _write_all(marker_fd, b"asi-cpr-reserved-v1\n")
        os.fsync(marker_fd)
        metadata = os.fstat(marker_fd)
        os.fsync(directory)
        reservation = directory, path.name, marker_name, marker_fd, metadata.st_dev, metadata.st_ino
        _require_live_parent(reservation)
        return reservation
    except BaseException:
        if marker_fd >= 0:
            try:
                metadata = os.fstat(marker_fd)
                visible = os.stat(marker_name, dir_fd=directory, follow_symlinks=False)
                if (visible.st_dev, visible.st_ino) == (metadata.st_dev, metadata.st_ino):
                    os.unlink(marker_name, dir_fd=directory)
                    os.fsync(directory)
            except FileNotFoundError:
                pass
            os.close(marker_fd)
        os.close(directory)
        raise


def _owned(reservation: Reservation) -> None:
    directory, _name, marker, marker_fd, device, inode = reservation
    held = os.fstat(marker_fd)
    visible = os.stat(marker, dir_fd=directory, follow_symlinks=False)
    if (
        not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or (held.st_dev, held.st_ino) != (device, inode)
        or (visible.st_dev, visible.st_ino) != (device, inode)
    ):
        raise RuntimeError("CPR output reservation identity changed")


def _finish_reservation(
    reservation: Reservation, *, consumed: bool, completed: bool = False
) -> None:
    directory, _name, marker, marker_fd, device, inode = reservation
    try:
        _owned(reservation)
        if consumed or completed:
            if completed:
                _require_live_parent(reservation)
            os.lseek(marker_fd, 0, os.SEEK_SET)
            data = (
                b"asi-cpr-completed-v2\n" if completed else b"asi-cpr-consumed-without-result-v1\n"
            )
            _write_all(marker_fd, data)
            os.ftruncate(marker_fd, len(data))
            os.fsync(marker_fd)
            os.fsync(directory)
            if completed:
                _require_live_parent(reservation)
        else:
            current = os.stat(marker, dir_fd=directory, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (device, inode):
                os.unlink(marker, dir_fd=directory)
                os.fsync(directory)
    finally:
        os.close(marker_fd)
        os.close(directory)


def _link_tmpfile(file_fd: int, directory_fd: int, name: str) -> None:
    linkat = ctypes.CDLL(None, use_errno=True).linkat
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    if linkat(file_fd, b"", directory_fd, os.fsencode(name), 0x1000) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), name)
        raise OSError(number, os.strerror(number), name)


def _publish(
    reservation: Reservation,
    report: dict[str, object],
    x: np.ndarray,
    y: np.ndarray,
    config: IPMNISTConfig,
    seeds: tuple[int, ...],
    capability: object,
) -> None:
    if capability is not _EXECUTION_CAPABILITY and capability is not _TEST_EXECUTION_CAPABILITY:
        raise RuntimeError("CPR publication capability is invalid")
    if capability is _EXECUTION_CAPABILITY:
        _require_authorized()
    production = _check_request(config, seeds)
    if production is not (capability is _EXECUTION_CAPABILITY):
        raise RuntimeError("CPR publication roster and capability differ")
    if not production:
        _require_test_namespace(OUTPUT_PATH)
    directory, name, _marker, _marker_fd, _device, _inode = reservation
    _owned(reservation)
    _require_live_parent(reservation)
    validate_report(report, x, y, config=config, seeds=seeds, reexecute=True)
    encoded = _canonical(report) + b"\n"
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("CPR report exceeds its byte bound")
    descriptor = os.open(".", os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC, 0o600, dir_fd=directory)
    published_identity: tuple[int, int] | None = None
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0 or written > len(view):
                raise OSError("CPR report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_nlink != 0
            or staged.st_size != len(encoded)
        ):
            raise ValueError("CPR staged report inode is invalid")
        os.lseek(descriptor, 0, os.SEEK_SET)
        staged_raw = os.read(descriptor, len(encoded) + 1)
        if staged_raw != encoded:
            raise ValueError("CPR staged report changed during bounded reread")

        def exact(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("CPR published JSON contains duplicate keys")
                result[key] = item
            return result

        try:
            staged_report = json.loads(staged_raw.decode("utf-8"), object_pairs_hook=exact)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("CPR staged report is not strict bounded JSON") from error
        validate_report(staged_report, x, y, config=config, seeds=seeds, reexecute=False)
        _owned(reservation)
        _require_live_parent(reservation)
        published_identity = (staged.st_dev, staged.st_ino)
        _link_tmpfile(descriptor, directory, name)
        _require_live_parent(reservation)
        metadata = os.fstat(descriptor)
        if metadata.st_nlink != 1 or (metadata.st_dev, metadata.st_ino) != published_identity:
            raise RuntimeError("CPR published inode identity drifted")
        os.fsync(directory)
        _require_live_parent(reservation)
        read_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            before = os.fstat(read_fd)
            raw = os.read(read_fd, len(encoded) + 1)
            after = os.fstat(read_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_dev, before.st_ino) != published_identity
                or stat.S_IMODE(before.st_mode) != 0o400
                or raw != encoded
                or (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise ValueError("CPR published report changed during bounded reread")

            try:
                reread = json.loads(raw.decode("utf-8"), object_pairs_hook=exact)
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                raise ValueError("CPR published report is not strict bounded JSON") from error
            validate_report(reread, x, y, config=config, seeds=seeds, reexecute=False)
            _owned(reservation)
            _require_live_parent(reservation)
            visible = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (visible.st_dev, visible.st_ino) != published_identity:
                raise RuntimeError("CPR published report inode changed")
        finally:
            os.close(read_fd)
    except BaseException:
        if published_identity is not None:
            try:
                visible = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (visible.st_dev, visible.st_ino) == published_identity:
                    os.unlink(name, dir_fd=directory)
                    os.fsync(directory)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(descriptor)


def run_and_publish(data_home: Path, destination: Path = OUTPUT_PATH) -> dict[str, object]:
    _require_authorized()
    return _run_and_publish(
        data_home, destination, CAMPAIGN_CONFIG, CAMPAIGN_SEEDS, _EXECUTION_CAPABILITY
    )


def _run_and_publish(
    data_home: Path,
    destination: Path,
    config: IPMNISTConfig,
    seeds: tuple[int, ...],
    capability: object,
) -> dict[str, object]:
    if capability is not _EXECUTION_CAPABILITY and capability is not _TEST_EXECUTION_CAPABILITY:
        raise RuntimeError("CPR transaction capability is invalid")
    if capability is _EXECUTION_CAPABILITY:
        _require_authorized()
    production = _check_request(config, seeds)
    if production is not (capability is _EXECUTION_CAPABILITY) or type(data_home) is not type(
        Path()
    ):
        raise RuntimeError("CPR transaction capability is invalid")
    if not production:
        _require_test_namespace(destination)
    reservation = _reserve(destination)
    dispatched = False
    published = False

    def note() -> None:
        nonlocal dispatched
        dispatched = True
        _mark_dispatched(reservation)

    try:
        _require_live_parent(reservation)
        if capability is _EXECUTION_CAPABILITY:
            _screening_source_provenance(_ROOT)
        x, y = load_mnist_train(data_home)
        report = _run(x, y, config=config, seeds=seeds, capability=capability, on_dispatch=note)
        _publish(reservation, report, x, y, config, seeds, capability)
        published = True
        return report
    finally:
        _finish_reservation(reservation, consumed=dispatched and not published, completed=published)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--data-home", type=Path)
    args = parser.parse_args(argv)
    if args.catalog:
        print(json.dumps(frozen_plan(), sort_keys=True))
        return 0
    _require_authorized()
    run_and_publish(args.data_home if args.data_home is not None else default_openml_data_home())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
