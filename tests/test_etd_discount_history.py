"""Public ETD updates follow the incoming discount in JMLR 2016, eqs. 17–20."""

from __future__ import annotations

import pickle

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.off_policy_td import ETDLinearLearner


@pytest.mark.parametrize("trace_decay", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("incoming", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("outgoing", [0.0, 0.5, 1.0])
def test_delayed_credit_and_emphasis_use_incoming_discount(
    trace_decay: float, incoming: float, outgoing: float
) -> None:
    learner = ETDLinearLearner(step_size=0.1, trace_decay=trace_decay)
    observations = jnp.eye(3, dtype=jnp.float32)
    first = learner.update(learner.init(3), observations[0], 0.0, observations[1], incoming, 2.0)
    second = learner.update(first.state, observations[1], 1.0, observations[2], outgoing, 0.5)
    assert bool(first.update_applied) and bool(second.update_applied)

    # First F and M are one, so e_0 = 2 * phi_0. No weights change until r_2.
    follow_on = 1.0 + 2.0 * incoming
    emphasis = trace_decay + (1.0 - trace_decay) * follow_on
    traces = np.array([incoming * trace_decay, 0.5 * emphasis, 0.0], dtype=np.float32)
    np.testing.assert_allclose(second.state.follow_on_trace, follow_on, rtol=1e-6)
    np.testing.assert_allclose(second.state.emphasis, emphasis, rtol=1e-6)
    np.testing.assert_allclose(second.state.eligibility_traces, traces, rtol=1e-6)
    np.testing.assert_allclose(second.state.weights, 0.1 * traces, rtol=1e-6)
    np.testing.assert_allclose(second.state.bias, 0.1 * traces.sum(), rtol=1e-6)


def test_terminal_boundary_isolates_the_next_stream() -> None:
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.4)
    observations = jnp.eye(3, dtype=jnp.float32)
    terminal = learner.update(learner.init(3), observations[0], 0.0, observations[1], 0.0, 3.0)
    following = learner.update(
        terminal.state, observations[1], 1.0, observations[2], 0.9, 1.0, interest=0.5
    )
    assert bool(terminal.update_applied) and bool(following.update_applied)
    np.testing.assert_allclose(following.state.follow_on_trace, 0.5)
    np.testing.assert_allclose(following.state.emphasis, 0.5)
    np.testing.assert_allclose(following.state.eligibility_traces, [0.0, 0.5, 0.0])
    np.testing.assert_allclose(following.state.weights, [0.0, 0.05, 0.0], rtol=1e-6)


def test_rejection_does_not_advance_discount_or_ratio_history() -> None:
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.4)
    observations = jnp.eye(3, dtype=jnp.float32)
    first = learner.update(learner.init(3), observations[0], 0.0, observations[1], 0.5, 2.0)
    rejected = learner.update(
        first.state, observations[1], jnp.float32(jnp.nan), observations[2], 0.0, 7.0
    )
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, first.state)
    continued = learner.update(rejected.state, observations[1], 1.0, observations[2], 0.0, 0.5)
    direct = learner.update(first.state, observations[1], 1.0, observations[2], 0.0, 0.5)
    chex.assert_trees_all_equal(continued, direct)


@pytest.mark.parametrize("discount", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_saved_discount_rejects_even_when_traces_are_zero(discount: float) -> None:
    learner = ETDLinearLearner(trace_decay=0.0)
    state = learner.init(2).replace(previous_gamma=jnp.float32(discount))
    result = learner.update(state, jnp.ones(2), 1.0, jnp.zeros(2), 0.0, 1.0)
    assert not bool(result.update_applied)
    for name, value in state.items():
        np.testing.assert_array_equal(result.state[name], value)


@pytest.mark.parametrize("history", [None, np.ones(1, dtype=np.float32), np.float32(0.5)])
def test_missing_or_noncanonical_saved_discount_is_not_inferred(history: object) -> None:
    learner = ETDLinearLearner()
    state = learner.init(2).replace(previous_gamma=history)
    with pytest.raises(ValueError, match="state.previous_gamma"):
        learner.update(state, jnp.ones(2), 1.0, jnp.zeros(2), 0.0, 1.0)


def test_terminal_bootstrap_does_not_discard_required_invalid_history() -> None:
    learner = ETDLinearLearner(trace_decay=0.4)
    state = learner.init(2).replace(follow_on_trace=jnp.float32(jnp.inf))
    result = learner.update(state, jnp.ones(2), 1.0, jnp.zeros(2), 0.0, 1.0)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_pickle_continuation_preserves_incoming_discount() -> None:
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.8)
    observations = jnp.eye(3, dtype=jnp.float32)
    first = learner.update(learner.init(3), observations[0], 0.0, observations[1], 0.5, 2.0)
    restored = pickle.loads(pickle.dumps(first.state))
    resumed = learner.update(restored, observations[1], 1.0, observations[2], 0.0, 0.5)
    direct = learner.update(first.state, observations[1], 1.0, observations[2], 0.0, 0.5)
    chex.assert_trees_all_equal(resumed, direct)
    np.testing.assert_allclose(resumed.state.weights, [0.04, 0.06, 0.0], rtol=1e-6)


def test_public_update_scan_retains_history_across_a_boundary() -> None:
    learner = ETDLinearLearner(step_size=0.1, trace_decay=0.8)
    observations = jnp.eye(3, dtype=jnp.float32)
    inputs = (
        observations,
        jnp.roll(observations, -1, axis=0),
        jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32),
        jnp.array([0.5, 0.0, 0.75], dtype=jnp.float32),
        jnp.array([2.0, 0.5, 1.0], dtype=jnp.float32),
    )
    initial = learner.init(3)

    def step(state, transition):
        observation, next_observation, reward, gamma, rho = transition
        result = learner.update(state, observation, reward, next_observation, gamma, rho)
        return result.state, result.update_applied

    scanned, applied = jax.lax.scan(step, initial, inputs)
    sequential = initial
    for transition in zip(*inputs, strict=True):
        sequential, accepted = step(sequential, transition)
        assert bool(accepted)
    assert bool(jnp.all(applied))
    chex.assert_trees_all_equal(scanned, sequential)
    np.testing.assert_array_equal(scanned.eligibility_traces, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(scanned.follow_on_trace, 1.0)


def test_numeric_state_accounts_for_the_saved_discount() -> None:
    state = ETDLinearLearner().init(3)
    payload = sum(value.nbytes for value in state.values() if isinstance(value, jax.Array))
    assert state.previous_gamma.shape == ()
    assert state.previous_gamma.dtype == jnp.float32
    assert state.previous_gamma.nbytes == 4
    assert payload == 4 * (2 * 3 + 7)
