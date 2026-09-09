"""Changing GVF discounts must preserve credit until the terminating update."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.multi_head_learner import measure_multi_head_mlp_state_nbytes
from alberta_framework.core.off_policy_horde import (
    OffPolicyHordeLearner,
    OffPolicyHordeState,
    run_off_policy_horde_learning_loop,
    run_off_policy_horde_learning_loop_batched,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, TraceMode, create_horde_spec


def _learner(mode=TraceMode.ACCUMULATING, n_demons=1):
    spec = create_horde_spec(
        tuple(
            GVFSpec(
                name=f"demon_{i}",
                demon_type=DemonType.PREDICTION,
                gamma=0.9,
                lamda=0.8,
                cumulant_index=i,
            )
            for i in range(n_demons)
        )
    )
    return OffPolicyHordeLearner(
        spec,
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        ratio_clip=10.0,
        trace_ratio_clip=10.0,
        trace_mode=mode,
    )


def _initial(learner):
    state = learner.init(3, jax.random.key(270, impl="threefry2x32"))
    return state.replace(head_params=jax.tree.map(jnp.zeros_like, state.head_params))


def _step(learner, state, index, rewards, discounts, rhos=None):
    return learner.update_with_ratios_and_discounts(
        state,
        jnp.eye(3)[index],
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.eye(3)[(index + 1) % 3],
        jnp.asarray(rhos if rhos is not None else [1.0] * learner.n_demons, dtype=jnp.float32),
        jnp.asarray(discounts, dtype=jnp.float32),
    )


@pytest.mark.parametrize("mode", list(TraceMode))
@pytest.mark.parametrize("incoming", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("outgoing", [0.0, 0.25, 1.0])
def test_delayed_credit_uses_incoming_discount(mode, incoming, outgoing):
    learner = _learner(mode)
    first = _step(learner, _initial(learner), 0, [0.0], [incoming], [2.0])
    result = _step(learner, first.state, 1, [1.0], [outgoing], [0.5])
    assert bool(first.update_applied) and bool(result.update_applied)
    # rho_1 gamma_1 lambda rho_0 phi_0 + rho_1 phi_1, with delta_1=1.
    expected = np.array([[0.1 * 0.5 * incoming * 0.8 * 2.0, 0.05, 0.0]])
    np.testing.assert_allclose(result.state.head_params.weights[0], expected, atol=1e-7)


@pytest.mark.parametrize("mode", list(TraceMode))
def test_terminal_credit_does_not_cross_into_next_episode(mode):
    learner = _learner(mode)
    first = _step(learner, _initial(learner), 0, [0.0], [0.5])
    terminal = _step(learner, first.state, 1, [1.0], [0.0])
    following = _step(learner, terminal.state, 2, [1.0], [0.9])
    np.testing.assert_array_equal(
        following.state.head_params.weights[0][:, :2],
        terminal.state.head_params.weights[0][:, :2],
    )


def _assert_learning_state_equal(left, right):
    for name in dict(left):
        if name in {"birth_timestamp", "uptime_s"}:
            continue
        for a, b in zip(
            jax.tree.leaves(getattr(left, name)), jax.tree.leaves(getattr(right, name)), strict=True
        ):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("skipped_reward", [np.nan, np.inf])
def test_inactive_or_rejected_head_preserves_its_own_discount(skipped_reward):
    learner = _learner(n_demons=2)
    first = _step(learner, _initial(learner), 0, [0.0, 0.0], [0.5, 0.9])
    skipped = _step(learner, first.state, 1, [skipped_reward, 0.0], [0.0, 0.25])
    np.testing.assert_array_equal(skipped.head_updates_applied, [False, True])
    np.testing.assert_array_equal(skipped.state.previous_discounts, [0.5, 0.25])
    np.testing.assert_array_equal(skipped.state.head_traces[0][0], first.state.head_traces[0][0])
    resumed = _step(learner, skipped.state, 2, [1.0, 1.0], [0.0, 0.0])
    # Head 0 resumes its last accepted trace; head 1 uses its newer history.
    np.testing.assert_allclose(resumed.state.head_params.weights[0], [[0.04, 0.0, 0.1]], atol=1e-7)
    np.testing.assert_allclose(
        resumed.state.head_params.weights[1], [[0.0144, 0.02, 0.1]], atol=1e-7
    )


def test_global_rejection_and_all_inactive_step_preserve_history():
    learner = _learner(n_demons=2)
    first = _step(learner, _initial(learner), 0, [0.0, 0.0], [0.5, 0.9])
    rejected = learner.update_with_ratios_and_discounts(
        first.state,
        jnp.full(3, jnp.inf),
        jnp.ones(2),
        jnp.zeros(3),
        jnp.ones(2),
        jnp.zeros(2),
    )
    assert not bool(rejected.update_applied)
    _assert_learning_state_equal(rejected.state, first.state)
    inactive = _step(learner, first.state, 1, [np.nan, np.nan], [0.0, 0.0])
    assert bool(inactive.update_applied)
    np.testing.assert_array_equal(inactive.state.previous_discounts, first.state.previous_discounts)
    assert int(inactive.state.step_count) == int(first.state.step_count) + 1


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_numeric_history_rejects_atomically(bad):
    learner = _learner()
    state = _initial(learner).replace(previous_discounts=jnp.array([bad], dtype=jnp.float32))
    rejected = _step(learner, state, 0, [1.0], [0.0])
    assert not bool(rejected.update_applied)
    _assert_learning_state_equal(rejected.state, state)


@pytest.mark.parametrize("bad", [None, jnp.array(0.5), jnp.ones(2), jnp.ones(1, dtype=jnp.int32)])
def test_missing_or_noncanonical_history_requires_explicit_adoption(bad):
    learner = _learner()
    state = _initial(learner).replace(previous_discounts=bad)
    with pytest.raises(ValueError, match="previous_discounts"):
        _step(learner, state, 0, [1.0], [0.0])


def test_legacy_state_requires_known_history_and_charges_only_one_float_per_demon():
    learner = _learner(n_demons=2)
    key = jax.random.key(270, impl="threefry2x32")
    legacy = learner.learner.init(3, key)
    with pytest.raises(ValueError, match="previous_discounts"):
        _step(learner, legacy, 0, [0.0, 0.0], [0.0, 0.0])
    adopted = OffPolicyHordeState(**dict(legacy), previous_discounts=jnp.ones(2))
    fresh = learner.init(3, key)
    _assert_learning_state_equal(adopted, fresh)
    assert (
        measure_multi_head_mlp_state_nbytes(fresh)
        == measure_multi_head_mlp_state_nbytes(legacy) + 8
    )
    result = _step(learner, adopted, 0, [0.0, 0.0], [0.0, 0.0])
    assert bool(result.update_applied)


@pytest.mark.parametrize("incoming, accepted", [(0.0, True), (0.5, False)])
def test_zero_outgoing_discount_does_not_hide_required_poisoned_trace(incoming, accepted):
    learner = _learner()
    first = _step(learner, _initial(learner), 0, [0.0], [incoming])
    state = first.state.replace(head_traces=((jnp.full((1, 3), jnp.inf), jnp.ones(1)),))
    result = _step(learner, state, 1, [1.0], [0.0])
    assert bool(result.update_applied) == accepted
    if accepted:
        np.testing.assert_array_equal(result.state.head_traces[0][0], [[0.0, 1.0, 0.0]])
    else:
        _assert_learning_state_equal(result.state, state)


def test_pickle_continuation_and_public_scan_preserve_discount_history():
    learner = _learner()
    initial = _initial(learner)
    first = _step(learner, initial, 0, [0.0], [0.5])
    restored = pickle.loads(pickle.dumps(first.state))
    expected = _step(learner, first.state, 1, [1.0], [0.0])
    actual = _step(learner, restored, 1, [1.0], [0.0])
    _assert_learning_state_equal(actual.state, expected.state)
    scanned = run_off_policy_horde_learning_loop(
        learner,
        initial,
        jnp.eye(3)[:2],
        jnp.array([[0.0], [1.0]]),
        jnp.eye(3)[1:],
        jnp.ones((2, 1)),
        jnp.array([[0.5], [0.0]]),
    )
    _assert_learning_state_equal(scanned.state, expected.state)
    np.testing.assert_array_equal(scanned.updates_applied, [True, True])


def test_batched_public_scan_keeps_history_in_its_state():
    learner = _learner()
    keys = jax.random.split(jax.random.key(270, impl="threefry2x32"), 2)
    result = run_off_policy_horde_learning_loop_batched(
        learner,
        jnp.eye(3)[:2],
        jnp.zeros((2, 1)),
        jnp.eye(3)[1:],
        jnp.ones((2, 1)),
        keys,
        jnp.array([[0.5], [0.0]]),
    )
    np.testing.assert_array_equal(result.state.previous_discounts, [[0.0], [0.0]])


def test_history_bytes_are_preflighted_before_wrapped_initialization(monkeypatch):
    import alberta_framework.core.off_policy_horde as module

    learner = _learner(n_demons=2)
    monkeypatch.setattr(module, "_multi_head_update_working_set_bytes", lambda *args: 2**31 - 20)

    def forbidden(*args):
        pytest.fail("oversized history envelope reached allocation")

    monkeypatch.setattr(learner.learner, "init", forbidden)
    with pytest.raises(ValueError, match="working set"):
        learner.init(3, jax.random.key(270, impl="threefry2x32"))
