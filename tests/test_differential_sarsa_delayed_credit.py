"""Delayed-reward credit must follow discounts between rewarded decisions."""

from __future__ import annotations

import dataclasses
import pickle

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    measure_differential_sarsa_state_nbytes,
    migrate_legacy_differential_sarsa_state,
    run_differential_sarsa_from_arrays,
)

pytestmark = pytest.mark.unit


def _agent(*, use_bias: bool = False) -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.1,
            average_reward_step_size=0.0,
            trace_decay=0.8,
            epsilon_start=0.0,
            use_bias=use_bias,
        )
    )


@pytest.mark.parametrize("incoming", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("outgoing", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("use_bias", [False, True])
def test_delayed_reward_uses_the_discount_entering_the_rewarded_decision(
    incoming: float, outgoing: float, use_bias: bool
) -> None:
    """Zero initialized public transitions give an independent analytic update."""
    agent = _agent(use_bias=use_bias)
    observations = jnp.eye(3, dtype=jnp.float32)
    state, _ = agent.start_with_action(
        agent.init(3, jr.key(250, impl="threefry2x32")), observations[0], jnp.int32(0)
    )
    first = agent.update(
        state, jnp.float32(0), observations[1], jnp.int32(1), discount=jnp.float32(incoming)
    )
    second = agent.update(
        first.state,
        jnp.float32(1),
        observations[2],
        jnp.int32(0),
        discount=jnp.float32(outgoing),
    )
    assert bool(first.update_applied) and bool(second.update_applied)
    assert float(first.td_error) == 0.0
    assert float(second.td_error) == 1.0
    expected = np.zeros((2, 3), dtype=np.float32)
    expected[0, 0] = 0.1 * incoming * 0.8
    expected[1, 1] = 0.1
    np.testing.assert_allclose(second.state.q_weights, expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(
        second.state.q_bias,
        np.array([0.1 * incoming * 0.8, 0.1]) * use_bias,
        atol=1e-7,
        rtol=1e-6,
    )


def test_zero_discount_cuts_credit_to_the_previous_stream() -> None:
    agent = _agent()
    observations = jnp.eye(3, dtype=jnp.float32)
    state, _ = agent.start_with_action(
        agent.init(3, jr.key(251, impl="threefry2x32")), observations[0], jnp.int32(0)
    )
    first = agent.update(state, jnp.float32(0), observations[1], jnp.int32(1), discount=1.0)
    boundary = agent.update(
        first.state, jnp.float32(1), observations[2], jnp.int32(0), discount=0.0
    )
    following = agent.update(
        boundary.state, jnp.float32(1), jnp.zeros(3), jnp.int32(1), discount=1.0
    )
    assert bool(following.update_applied)
    chex.assert_trees_all_equal(following.state.q_weights[:, :2], boundary.state.q_weights[:, :2])
    np.testing.assert_allclose(following.state.q_weights[0, 2], 0.1)


def test_rejected_transition_and_pickle_resume_preserve_pending_credit() -> None:
    agent = _agent()
    observations = jnp.eye(3, dtype=jnp.float32)
    state, _ = agent.start_with_action(
        agent.init(3, jr.key(252, impl="threefry2x32")), observations[0], jnp.int32(0)
    )
    first = agent.update(state, jnp.float32(0), observations[1], jnp.int32(1), discount=0.25)
    rejected = agent.update(
        first.state, jnp.float32(jnp.nan), observations[2], jnp.int32(0), discount=0.0
    )
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, first.state)
    restored = pickle.loads(pickle.dumps(rejected.state))
    direct = agent.update(first.state, jnp.float32(1), observations[2], jnp.int32(0), discount=0.0)
    resumed = agent.update(restored, jnp.float32(1), observations[2], jnp.int32(0), discount=0.0)
    chex.assert_trees_all_equal(direct, resumed)
    np.testing.assert_allclose(resumed.state.q_weights[0, 0], 0.02)


def test_array_runner_keeps_discount_history_and_boundary_credit() -> None:
    agent = DifferentialSARSAAgent(dataclasses.replace(_agent().config, n_actions=1))
    observations = jnp.eye(3, dtype=jnp.float32)
    state, _ = agent.start(agent.init(3, jr.key(253, impl="threefry2x32")), observations[0])
    result = run_differential_sarsa_from_arrays(
        agent,
        state,
        jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32),
        jnp.stack([observations[1], observations[2], jnp.zeros(3)]),
        discounts=jnp.array([0.25, 0.0, 1.0], dtype=jnp.float32),
    )
    assert bool(jnp.all(result.updates_applied))
    np.testing.assert_allclose(result.state.q_weights, [[0.02, 0.1, 0.1]], rtol=1e-6)


@pytest.mark.parametrize("discount", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_saved_discount_rejects_the_whole_update(discount: float) -> None:
    agent = _agent()
    state, _ = agent.start(agent.init(1, jr.key(254, impl="threefry2x32")), jnp.ones(1))
    # Convert the pre-existing host telemetry to the compiled state dtype first.
    state = agent.update(state, jnp.float32(0), jnp.ones(1)).state
    poisoned = state.replace(previous_discount=jnp.float32(discount))
    result = jax.jit(agent.update)(poisoned, jnp.float32(1), jnp.ones(1), discount=0.0)
    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    for before, after in zip(jax.tree.leaves(poisoned), jax.tree.leaves(result.state), strict=True):
        if jax.dtypes.issubdtype(before.dtype, jax.dtypes.prng_key):
            before, after = jr.key_data(before), jr.key_data(after)
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("has_exact_clock", [False, True])
def test_legacy_migration_requires_history_for_active_traces(has_exact_clock: bool) -> None:
    agent = _agent()
    state, _ = agent.start(agent.init(1, jr.key(255, impl="threefry2x32")), jnp.ones(1))
    state = agent.update(state, jnp.float32(0), jnp.ones(1), discount=0.25).state
    omitted = {"previous_discount"} | (set() if has_exact_clock else {"step_words"})
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name not in omitted
    }
    with pytest.raises(ValueError, match="explicit previous_discount"):
        migrate_legacy_differential_sarsa_state(legacy)
    migrated = migrate_legacy_differential_sarsa_state(legacy, previous_discount=0.25)
    chex.assert_trees_all_equal(state, migrated)
    with pytest.raises(ValueError, match="previous_discount"):
        migrate_legacy_differential_sarsa_state(legacy, previous_discount=1.1)
    with pytest.raises(ValueError, match="state schema"):
        DifferentialSARSAAgent.from_config(
            {**agent.to_config(), "state_schema": "alberta.differential-sarsa-state.v2"}
        )


def test_discount_cursor_adds_one_accounted_float32_scalar() -> None:
    state = _agent().init(3, jr.key(256, impl="threefry2x32"))
    assert state.previous_discount.shape == ()
    assert state.previous_discount.dtype == jnp.float32
    # Two Q/trace matrices, two bias/trace vectors, observation, reward rate,
    # epsilon, the discount, two int32 scalars and four uint32 key/clock words.
    assert measure_differential_sarsa_state_nbytes(state) == 4 * (2 * 2 * 3 + 2 * 2 + 3 + 3 + 6)
