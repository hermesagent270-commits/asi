"""Dataset-bound replay contracts using the existing eight-update fixture."""

import dataclasses
import json

import numpy as np
import pytest

from alberta_framework.benchmarks import nap_ipmnist as nap

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def case():
    images = (np.arange(12 * 784, dtype=np.float32).reshape(12, 784) % 256) / 255.0
    labels = np.arange(12, dtype=np.int32) % 10
    result = nap.run_comparator(images, labels, seed=nap.FROZEN_SEEDS[0])
    return result, images, labels


def forge(result, field):
    arm = result.arms[-1]
    if field == "task_loss":
        arm = dataclasses.replace(arm, task_loss=(arm.task_loss[0] + 0.125, *arm.task_loss[1:]))
    elif field == "final_state_sha256":
        arm = dataclasses.replace(arm, final_state_sha256="0" * 64)
    else:
        value = 0.0 if arm.dead_unit_fraction[0] > 0.5 else 1.0
        arm = dataclasses.replace(arm, dead_unit_fraction=(value, *arm.dead_unit_fraction[1:]))
    return dataclasses.replace(result, arms=(*result.arms[:-1], arm))


@pytest.mark.parametrize("field", ["task_loss", "final_state_sha256", "dead_unit_fraction"])
def test_finite_forged_fields_fail_independent_reexecution(case, field):
    result, images, labels = case
    forged = forge(result, field)
    # This is the live structural-validation hole, rather than an invalid type
    # or receipt that the existing validator already rejects.
    assert nap.validate_result(forged) is forged
    with pytest.raises(ValueError):
        nap.validate_result_with_dataset(forged, images, labels)


def test_json_roundtrip_and_exact_reexecution_ignore_only_timing(case):
    result, images, labels = case
    encoded = nap._json_result(result)
    restored = nap.result_from_json(encoded)
    assert nap._json_result(restored) == encoded
    retimed = dataclasses.replace(
        restored,
        arms=tuple(
            dataclasses.replace(arm, receipt=dataclasses.replace(arm.receipt, elapsed_ns=0))
            for arm in restored.arms
        ),
    )
    replayed = nap.validate_result_with_dataset(retimed, images, labels)
    for expected, actual in zip(result.arms, replayed.arms, strict=True):
        assert dataclasses.replace(
            expected, receipt=dataclasses.replace(expected.receipt, elapsed_ns=0)
        ) == dataclasses.replace(actual, receipt=dataclasses.replace(actual.receipt, elapsed_ns=0))
        assert actual.receipt.elapsed_ns > 0


@pytest.mark.parametrize("axis", ["dataset", "schedule"])
def test_binding_mismatch_stops_before_learner_dispatch(case, monkeypatch, axis):
    result, images, labels = case

    def forbidden(*args, **kwargs):
        pytest.fail("binding mismatch dispatched a learner")

    monkeypatch.setattr(nap, "run_comparator", forbidden)
    if axis == "dataset":
        labels = labels.copy()
        labels[0] = (labels[0] + 1) % 10
    else:
        result = dataclasses.replace(result, schedule_sha256="0" * 64)
    with pytest.raises(ValueError):
        nap.validate_result_with_dataset(result, images, labels)


@pytest.mark.parametrize(
    "fault", ["unknown_field", "duplicate_key", "nonfinite", "wrong_curve_shape", "integer_flag"]
)
def test_json_codec_rejects_malformed_records(case, fault):
    result, _, _ = case
    payload = json.loads(nap._json_result(result))
    if fault == "duplicate_key":
        encoded = '{"schema":"duplicate",' + nap._json_result(result)[1:]
    elif fault == "nonfinite":
        payload["arms"][0]["task_loss"][0] = float("nan")
        encoded = json.dumps(payload)
    else:
        if fault == "unknown_field":
            payload["unbound"] = True
        elif fault == "integer_flag":
            payload["arms"][0]["normalization_enabled"] = 0
        else:
            payload["arms"][0]["task_loss"] *= 100
        encoded = json.dumps(payload)
    with pytest.raises(ValueError):
        nap.result_from_json(encoded)


def test_profile_curve_bound_rejects_before_arm_construction(case, monkeypatch):
    result, _, _ = case
    payload = json.loads(nap._json_result(result))
    for name in ("task_accuracy", "task_loss", "dead_unit_fraction", "effective_rank"):
        payload["arms"][0][name].append(payload["arms"][0][name][-1])

    def forbidden(_self):
        pytest.fail("a curve larger than its profile reached arm construction")

    monkeypatch.setattr(nap.NaPArmResult, "__post_init__", forbidden)
    with pytest.raises(ValueError, match="sequence"):
        nap.result_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_id", "no-such-profile"),
        ("profile_id", ["contract-smoke"]),
        ("seed", "x"),
        ("seed", True),
        ("seed", 0),
        ("profile", "registry-mismatch"),
    ],
)
def test_json_admission_rejects_invalid_profile_or_seed(case, monkeypatch, field, value):
    result, _, _ = case
    payload = json.loads(nap._json_result(result))
    if field == "profile":
        payload["profile"]["n_tasks"] += 1
    else:
        payload[field] = value

    def forbidden(*args, **kwargs):
        pytest.fail("invalid JSON admission dispatched a learner")

    monkeypatch.setattr(nap, "run_comparator", forbidden)
    with pytest.raises(ValueError):
        nap.result_from_json(json.dumps(payload))


@pytest.mark.parametrize("forged", [False, True])
def test_cli_reloads_and_validates_saved_result(case, tmp_path, capsys, forged):
    result, images, labels = case
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, images=images, labels=labels)
    artifact = tmp_path / "result.json"
    artifact.write_text(nap._json_result(forge(result, "task_loss") if forged else result))
    argv = ["--dataset", str(dataset), "--validate", str(artifact)]
    if forged:
        with pytest.raises(ValueError):
            nap.main(argv)
    else:
        assert nap.main(argv) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["validated"] is True
        assert report["scientific_promotion_allowed"] is False
        assert report["seed"] == result.seed
        receipt = report["validation_receipt"]
        assert receipt["arm_reexecutions"] == 5
        assert receipt["data_steps"] == 40
        assert receipt["model_queries"] == 120
        assert receipt["parameter_updates"] == 40
        assert len(receipt["per_arm_receipts"]) == 5
        assert receipt["timing_telemetry_only"] is True
    # Validation emits a receipt; it never rewrites the supplied record.
    assert artifact.read_text() == nap._json_result(
        forge(result, "task_loss") if forged else result
    )
