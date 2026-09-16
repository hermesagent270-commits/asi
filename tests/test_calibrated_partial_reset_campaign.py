from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import jax
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.evaluation.calibrated_partial_reset_campaign as lane
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

SMALL = IPMNISTConfig(n_tasks=1, task_length=8, input_dim=4, hidden1=3, hidden2=2, n_classes=2)


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(32, dtype=np.float32).reshape(8, 4) / 32.0,
        np.arange(8, dtype=np.int32) % 2,
    )


def _fake_run(
    data_x: np.ndarray,
    data_y: np.ndarray,
    spec: object,
    seed: int,
    config: IPMNISTConfig,
) -> ScreeningRunResult:
    del data_x, data_y
    checked = cast(Any, spec)
    value = 0.4 + 0.01 * (lane.TEST_ONLY_SEEDS.index(seed) + lane.ARMS.index(checked.name))
    return ScreeningRunResult(
        config_name=checked.name,
        base_learner=checked.base_learner,
        hyperparameters=checked.hyperparameters,
        seed=seed,
        config=config,
        per_task_accuracy=np.full(config.n_tasks, value, dtype=np.float64),
        per_task_loss=np.full(config.n_tasks, 1.0 - value, dtype=np.float64),
        per_task_plasticity=np.full(config.n_tasks, value, dtype=np.float64),
        wall_clock_seconds=1.0,
    )


def _freeze_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake-runner transactions use one fixed runtime fixture. The actual
    # distribution/config inventory has its own direct regression below.
    runtime = lane._runtime_identity()
    monkeypatch.setattr(lane, "_runtime_identity", lambda: copy.deepcopy(runtime))
    source = lane._source_identity()
    monkeypatch.setattr(lane, "_source_identity", lambda: dict(source))


def _run_for_test(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    _freeze_runtime(monkeypatch)
    monkeypatch.setattr(lane, "run_screening_config", _fake_run)
    return lane._run(
        *_data(),
        config=SMALL,
        seeds=lane.TEST_ONLY_SEEDS,
        capability=lane._TEST_EXECUTION_CAPABILITY,
    )


def _resign(report: dict[str, object]) -> None:
    unsigned = dict(report)
    unsigned.pop("sha256", None)
    report["sha256"] = hashlib.sha256(lane._canonical(unsigned)).hexdigest()


def test_plan_is_prospective_exact_and_nonpromoting() -> None:
    plan = lane.frozen_plan()
    assert plan["seeds"] == list(lane.CAMPAIGN_SEEDS)
    assert set(plan["seeds"]).isdisjoint(plan["test_only_seeds"])
    assert plan["reviewed_execution_transition"] is False
    assert plan["execution_authorized"] is False
    assert plan["scientific_promotion_allowed"] is False
    assert plan["arms"] == list(lane.ARMS)
    assert "pyproject.toml" in lane._source_identity()
    assert "uv.lock" in lane._source_identity()
    assert plan["resources"]["combined_numeric_bytes"] <= 256 * 1024 * 1024
    assert plan["transaction_resources"] == {
        "campaign_rows": 25,
        "initial_runner_dispatches": 25,
        "strict_reexecution_dispatches": 25,
        "total_runner_dispatches": 50,
        "total_observations": 15_000_000,
        "total_updates": 15_000_000,
        "total_data_steps": 15_000_000,
        "total_environment_steps": 0,
        "total_model_queries": 30_000_000,
    }
    runtime = lane._runtime_identity()
    assert runtime["jax"]["config"]["jax_default_prng_impl"] == "threefry2x32"
    assert runtime["jax"]["config"]["jax_random_seed_offset"] == 0


def test_public_transaction_is_closed_before_reservation_or_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("consumer or reservation ran before authorization")

    monkeypatch.setattr(lane, "_reserve", forbidden)
    monkeypatch.setattr(lane, "load_mnist_train", forbidden)
    monkeypatch.setattr(lane, "run_screening_config", forbidden)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.run_and_publish(tmp_path, tmp_path / "report.json")
    assert calls == 0


def test_private_runner_covers_complete_roster_and_strict_reexecution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_for_test(monkeypatch)
    assert [(row["seed"], row["arm"]) for row in report["rows"]] == [
        (seed, arm) for seed in lane.TEST_ONLY_SEEDS for arm in lane.ARMS
    ]
    primary = report["aggregate"]["primary_paired_question"]
    assert [item["seed"] for item in primary["paired_deltas"]] == list(lane.TEST_ONLY_SEEDS)
    assert primary["mean_delta"] == pytest.approx(0.01)
    assert primary["positive_seed_count"] == 5
    assert primary["outcome"] == "advance_for_nonpromoting_followup"
    lane.validate_report(
        report,
        *_data(),
        config=SMALL,
        seeds=lane.TEST_ONLY_SEEDS,
        reexecute=True,
    )


def test_validator_rejects_resource_and_aggregate_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_for_test(monkeypatch)
    forged = copy.deepcopy(report)
    forged["rows"][0]["result"]["resources"]["updates"] = 1
    _resign(forged)
    with pytest.raises(ValueError, match="resource"):
        lane.validate_report(
            forged,
            *_data(),
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            reexecute=False,
        )
    forged = copy.deepcopy(report)
    forged["aggregate"]["row_count"] = 1
    _resign(forged)
    with pytest.raises(ValueError, match="aggregate"):
        lane.validate_report(
            forged,
            *_data(),
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            reexecute=False,
        )


def test_json_boundary_rejects_hostile_nested_type_without_hooks() -> None:
    calls = 0

    class Meta(type):
        def __hash__(cls) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile type hash dispatched")

        def __eq__(cls, other: object) -> bool:
            del other
            nonlocal calls
            calls += 1
            raise AssertionError("hostile type equality dispatched")

    class Hostile(metaclass=Meta):
        pass

    with pytest.raises(ValueError, match="exact JSON"):
        lane._bounded_json({"nested": [Hostile()]})
    assert calls == 0


def test_runtime_device_bound_precedes_device_attribute_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Device:
        def __getattribute__(self, name: str) -> object:
            del name
            nonlocal calls
            calls += 1
            raise AssertionError("device attribute accessed before inventory bound")

    monkeypatch.setattr(lane.jax, "devices", lambda: [Device()] * 65)
    with pytest.raises(RuntimeError, match="inventory"):
        lane._runtime_identity()
    assert calls == 0


def test_json_boundary_rejects_unbounded_integer() -> None:
    assert lane._bounded_json({"values": [-(2**63), 2**63 - 1]}) == {
        "values": [-(2**63), 2**63 - 1]
    }
    for value in (-(2**63) - 1, 2**63):
        with pytest.raises(ValueError, match="out-of-bounds integer"):
            lane._bounded_json({"nested": [value]})


def test_json_boundary_enforces_aggregate_utf8_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lane, "_MAX_TOTAL_UTF8_BYTES", 7)
    with pytest.raises(ValueError, match="UTF-8|oversized string"):
        lane._canonical({"a": "1234", "b": "5678"})


def test_validator_rejects_bool_int_aliases_in_nested_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_for_test(monkeypatch)
    forged = copy.deepcopy(report)
    forged["plan"]["execution_authorized"] = 0
    _resign(forged)
    with pytest.raises(ValueError, match="plan"):
        lane.validate_report(
            forged,
            *_data(),
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            reexecute=False,
        )
    forged = copy.deepcopy(report)
    forged["policy"]["development_only"] = 1
    _resign(forged)
    with pytest.raises(ValueError, match="policy"):
        lane.validate_report(
            forged,
            *_data(),
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            reexecute=False,
        )


def test_combined_numeric_bound_precedes_dataset_copy_and_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _data()
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("allocation or runner preceded aggregate bound")

    monkeypatch.setattr(lane, "_MAX_NUMERIC_BYTES", 1)
    monkeypatch.setattr(lane.np, "array", forbidden)
    monkeypatch.setattr(lane, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="combined numeric allocation"):
        lane._run(
            x,
            y,
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            capability=lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_noncontiguous_dataset_is_rejected_before_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = _data()
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("runner preceded dataset allocation validation")

    monkeypatch.setattr(lane, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="C-contiguous"):
        lane._run(
            x[:, ::-1],
            y,
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            capability=lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_transaction_reserves_before_load_and_retains_tombstone_after_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _freeze_runtime(monkeypatch)
    destination = tmp_path / "report.json"
    marker = destination.with_name(f".{destination.name}.reservation")
    marker.write_bytes(b"occupied")
    calls = 0

    def fail(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("consumer failure")

    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane, "load_mnist_train", fail)
    with pytest.raises(FileExistsError):
        lane._run_and_publish(
            tmp_path,
            destination,
            SMALL,
            lane.TEST_ONLY_SEEDS,
            lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0
    marker.unlink()
    monkeypatch.setattr(lane, "load_mnist_train", lambda _home: _data())
    monkeypatch.setattr(lane, "run_screening_config", fail)
    with pytest.raises(RuntimeError, match="consumer failure"):
        lane._run_and_publish(
            tmp_path,
            destination,
            SMALL,
            lane.TEST_ONLY_SEEDS,
            lane._TEST_EXECUTION_CAPABILITY,
        )
    assert marker.read_bytes() == b"asi-cpr-consumed-without-result-v1\n"
    with pytest.raises(FileExistsError):
        lane._run_and_publish(
            tmp_path,
            destination,
            SMALL,
            lane.TEST_ONLY_SEEDS,
            lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 1


def test_transaction_strictly_publishes_and_retains_completion_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _freeze_runtime(monkeypatch)
    destination = tmp_path / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane, "load_mnist_train", lambda _home: _data())
    monkeypatch.setattr(lane, "run_screening_config", _fake_run)
    report = lane._run_and_publish(
        tmp_path,
        destination,
        SMALL,
        lane.TEST_ONLY_SEEDS,
        lane._TEST_EXECUTION_CAPABILITY,
    )
    assert json.loads(destination.read_bytes()) == report
    marker = destination.with_name(f".{destination.name}.reservation")
    assert marker.read_bytes() == b"asi-cpr-completed-v2\n"
    destination.unlink()
    with pytest.raises(FileExistsError):
        lane._run_and_publish(
            tmp_path, destination, SMALL, lane.TEST_ONLY_SEEDS, lane._TEST_EXECUTION_CAPABILITY
        )


@pytest.mark.parametrize("gates", [(False, False), (True, False), (False, True), (1, True)])
def test_campaign_reexecution_gate_precedes_every_consumer(
    monkeypatch: pytest.MonkeyPatch, gates: tuple[object, object]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("campaign work preceded authorization")

    monkeypatch.setattr(lane, "_REVIEWED_EXECUTION_TRANSITION", gates[0])
    monkeypatch.setattr(lane, "_EXECUTION_AUTHORIZED", gates[1])
    for name in (
        "_validated_arrays",
        "_source_identity",
        "_runtime_identity",
        "_execution_identity",
        "run_screening_config",
    ):
        monkeypatch.setattr(lane, name, forbidden)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.validate_report(
            None,
            object(),
            object(),
            config=lane.CAMPAIGN_CONFIG,
            seeds=lane.CAMPAIGN_SEEDS,
            reexecute=True,
        )


def _replace_parent(destination: Path) -> Path:
    hidden = destination.parent.with_name("displaced")
    destination.parent.rename(hidden)
    destination.parent.mkdir()
    return hidden


def test_replaced_parent_after_reservation_blocks_dataset_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "registered" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    reserve = lane._reserve

    def replaced(path: Path) -> lane.Reservation:
        reservation = reserve(path)
        _replace_parent(destination)
        return reservation

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("dataset load preceded live-parent check")

    monkeypatch.setattr(lane, "_reserve", replaced)
    monkeypatch.setattr(lane, "load_mnist_train", forbidden)
    with pytest.raises(RuntimeError, match="parent"):
        lane._run_and_publish(
            tmp_path, destination, SMALL, lane.TEST_ONLY_SEEDS, lane._TEST_EXECUTION_CAPABILITY
        )


@pytest.mark.parametrize("stage", ["link", "directory_fsync"])
def test_replaced_parent_during_publication_rejects_hidden_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    _freeze_runtime(monkeypatch)
    destination = tmp_path / "registered" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane, "load_mnist_train", lambda _home: _data())
    monkeypatch.setattr(lane, "run_screening_config", _fake_run)
    link = lane._link_tmpfile
    fsync = os.fsync
    swapped = False

    def swap_link(file_fd: int, directory_fd: int, name: str) -> None:
        nonlocal swapped
        if stage == "link":
            _replace_parent(destination)
            swapped = True
        link(file_fd, directory_fd, name)

    def swap_fsync(descriptor: int) -> None:
        nonlocal swapped
        fsync(descriptor)
        if (
            stage == "directory_fsync"
            and not swapped
            and destination.exists()
            and os.fstat(descriptor).st_ino == destination.parent.stat().st_ino
        ):
            _replace_parent(destination)
            swapped = True

    monkeypatch.setattr(lane, "_link_tmpfile", swap_link)
    monkeypatch.setattr(lane.os, "fsync", swap_fsync)
    with pytest.raises(RuntimeError, match="parent"):
        lane._run_and_publish(
            tmp_path, destination, SMALL, lane.TEST_ONLY_SEEDS, lane._TEST_EXECUTION_CAPABILITY
        )
    assert swapped
    assert not destination.exists()
    assert not (tmp_path / "displaced" / destination.name).exists()


@pytest.mark.parametrize("stage", ["reservation", "tombstone"])
def test_marker_short_writes_retain_every_byte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    destination = tmp_path / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    write = os.write

    def short(descriptor: int, data: bytes) -> int:
        return write(descriptor, data[:1])

    if stage == "reservation":
        monkeypatch.setattr(lane.os, "write", short)
    reservation = lane._reserve(destination)
    marker = destination.with_name(f".{destination.name}.reservation")
    try:
        assert marker.read_bytes() == b"asi-cpr-reserved-v1\n"
    finally:
        if stage == "tombstone":
            monkeypatch.setattr(lane.os, "write", short)
        lane._finish_reservation(reservation, consumed=stage == "tombstone")
    if stage == "tombstone":
        assert marker.read_bytes() == b"asi-cpr-consumed-without-result-v1\n"


def test_zero_progress_reservation_write_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane.os, "write", lambda _fd, _data: 0)
    with pytest.raises(OSError, match="progress"):
        lane._reserve(destination)


def test_cli_gate_precedes_default_dataset_home_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> Path:
        raise AssertionError("dataset home discovery preceded authorization")

    monkeypatch.setattr(lane, "default_openml_data_home", forbidden)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.main([])


def test_gate_flip_without_reviewed_plan_digest_blocks_before_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("reservation preceded plan verification")

    monkeypatch.setattr(lane, "_REVIEWED_EXECUTION_TRANSITION", True)
    monkeypatch.setattr(lane, "_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(lane, "_reserve", forbidden)
    with pytest.raises(RuntimeError, match="digest drifted"):
        lane.run_and_publish(tmp_path)


@pytest.mark.parametrize("forgery", ["digest", "last_row", "aggregate"])
def test_complete_structural_validation_precedes_reexecution(
    monkeypatch: pytest.MonkeyPatch, forgery: str
) -> None:
    report = _run_for_test(monkeypatch)
    if forgery == "digest":
        report["sha256"] = "0" * 64
    elif forgery == "last_row":
        report["rows"][-1]["result"]["resources"]["updates"] = 0
        _resign(report)
    else:
        report["aggregate"]["row_count"] = 0
        _resign(report)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("runner preceded complete structural validation")

    monkeypatch.setattr(lane, "run_screening_config", forbidden)
    with pytest.raises(ValueError):
        lane.validate_report(
            report, *_data(), config=SMALL, seeds=lane.TEST_ONLY_SEEDS, reexecute=True
        )


def test_dataset_reexecution_rejects_self_consistent_changed_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_for_test(monkeypatch)
    report["rows"][0]["result"]["metrics"]["per_task_accuracy"][0] = 0.25
    report["aggregate"] = lane._aggregate(report["rows"])
    _resign(report)
    lane.validate_report(
        report, *_data(), config=SMALL, seeds=lane.TEST_ONLY_SEEDS, reexecute=False
    )
    with pytest.raises(ValueError, match="strict current reexecution"):
        lane.validate_report(
            report, *_data(), config=SMALL, seeds=lane.TEST_ONLY_SEEDS, reexecute=True
        )


def test_persistent_payload_matches_actual_authoritative_learner_states() -> None:
    params = lane.init_mlp_params(jr.key(301, impl="threefry2x32"), SMALL)
    for arm in lane.ARMS:
        spec = lane.screening_spec(arm)
        init_fn, _ = spec.factory(spec.hyperparameters)
        state = init_fn(params)
        payload_bytes = sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves((params, state)))
        assert payload_bytes == lane._resource_envelope(SMALL, 8)["peak_persistent_numeric_bytes"]


def test_dispatch_marker_is_durable_before_first_learner_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _freeze_runtime(monkeypatch)
    destination = tmp_path / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    monkeypatch.setattr(lane, "load_mnist_train", lambda _home: _data())
    marker = destination.with_name(f".{destination.name}.reservation")
    fsync = os.fsync
    synced_inodes = set()

    def tracked(descriptor: int) -> None:
        fsync(descriptor)
        synced_inodes.add(os.fstat(descriptor).st_ino)

    def failed(*args: object, **kwargs: object) -> object:
        assert marker.read_bytes() == b"asi-cpr-dispatched-consumed-v2\n"
        assert marker.stat().st_ino in synced_inodes
        assert destination.parent.stat().st_ino in synced_inodes
        raise RuntimeError("first learner failure")

    monkeypatch.setattr(lane.os, "fsync", tracked)
    monkeypatch.setattr(lane, "run_screening_config", failed)
    with pytest.raises(RuntimeError, match="first learner failure"):
        lane._run_and_publish(
            tmp_path, destination, SMALL, lane.TEST_ONLY_SEEDS, lane._TEST_EXECUTION_CAPABILITY
        )
    assert marker.read_bytes() == b"asi-cpr-consumed-without-result-v1\n"


@pytest.mark.parametrize(
    ("deltas", "outcome"),
    [
        ([0.006] * 5, "advance_for_nonpromoting_followup"),
        ([0.005] * 5, "inconclusive"),
        ([0.0] * 5, "do_not_advance"),
        ([-0.001] * 5, "do_not_advance"),
        ([0.008, 0.008, 0.008, 0.008, 0.0], "inconclusive"),
    ],
)
def test_primary_gate_requires_margin_and_every_seed_without_ablation_rescue(
    monkeypatch: pytest.MonkeyPatch, deltas: list[float], outcome: str
) -> None:
    report = _run_for_test(monkeypatch)
    for row in report["rows"]:
        if row["arm"] == "cpr_off":
            value = 0.5 if min(deltas) < 0 else 0.0
        elif row["arm"] == "cpr_ipmnist":
            value = deltas[lane.TEST_ONLY_SEEDS.index(row["seed"])]
            if min(deltas) < 0:
                value += 0.5
        else:
            value = 1.0
        row["result"]["metrics"]["per_task_accuracy"] = [value]
    aggregate = lane._aggregate(report["rows"])
    assert aggregate["outcome"] == outcome
    primary = aggregate["primary_paired_question"]
    observed = [item["utility_minus_off"] for item in primary["paired_deltas"]]
    assert primary["sample_standard_deviation"] == pytest.approx(np.std(observed, ddof=1))
    assert primary["standard_error"] == pytest.approx(np.std(observed, ddof=1) / np.sqrt(5))


def test_dataset_load_failure_releases_only_predispatch_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)

    def failed(_home: Path) -> object:
        raise RuntimeError("dataset failure before dispatch")

    monkeypatch.setattr(lane, "load_mnist_train", failed)
    with pytest.raises(RuntimeError, match="dataset failure"):
        lane._run_and_publish(
            tmp_path, destination, SMALL, lane.TEST_ONLY_SEEDS, lane._TEST_EXECUTION_CAPABILITY
        )
    assert not destination.with_name(f".{destination.name}.reservation").exists()


def test_runtime_binds_transitive_dependencies_and_complete_jax_config() -> None:
    runtime = lane._runtime_identity()
    assert {"ml-dtypes", "opt-einsum", "jax", "jaxlib", "numpy"} <= set(runtime["packages"])
    assert set(runtime["jax"]["all_config_values"]) == set(lane.jax.config.values)


@pytest.mark.parametrize("alias", ["counter", "timing_policy", "nested_policy"])
def test_campaign_rejects_nested_numeric_type_aliases_before_replay(
    monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    report = _run_for_test(monkeypatch)
    result = report["rows"][-1]["result"]
    if alias == "counter":
        result["resources"]["updates"] = float(SMALL.n_steps)
    elif alias == "timing_policy":
        result["resources"]["timing_is_selection_metric"] = 0
    else:
        result["policy"]["development_only"] = 1
    _resign(report)
    with pytest.raises(ValueError, match="exact nested"):
        lane.validate_report(
            report, *_data(), config=SMALL, seeds=lane.TEST_ONLY_SEEDS, reexecute=False
        )


@pytest.mark.parametrize("identity", ["_source_identity", "_runtime_identity", "_dataset_identity"])
def test_identity_drift_after_first_consumer_rejects_the_whole_report(
    monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    _freeze_runtime(monkeypatch)
    original = getattr(lane, identity)
    drifted = False
    calls = 0

    def changed(*args: object) -> object:
        return {"changed_identity": True} if drifted else original(*args)

    def consumer(*args: Any) -> ScreeningRunResult:
        nonlocal drifted, calls
        calls += 1
        result = _fake_run(*args)
        drifted = True
        return result

    monkeypatch.setattr(lane, identity, changed)
    monkeypatch.setattr(lane, "run_screening_config", consumer)
    with pytest.raises(RuntimeError, match="changed during execution"):
        lane._run(
            *_data(),
            config=SMALL,
            seeds=lane.TEST_ONLY_SEEDS,
            capability=lane._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 1


def test_test_only_capability_cannot_reserve_production_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("test-only capability reached production reservation")

    monkeypatch.setattr(lane, "_reserve", forbidden)
    with pytest.raises(RuntimeError, match="test-only namespace"):
        lane._run_and_publish(
            tmp_path, lane.OUTPUT_PATH, SMALL, lane.TEST_ONLY_SEEDS, lane._TEST_EXECUTION_CAPABILITY
        )
