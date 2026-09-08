"""Delayed reward credit uses the discount entering the current RTU state."""

import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticAgent,
    RecurrentTraceActorCriticConfig,
)


@pytest.fixture(scope="module")
def agent() -> RecurrentTraceActorCriticAgent:
    return RecurrentTraceActorCriticAgent(
        RecurrentTraceActorCriticConfig(
            n_actions=2,
            hidden_size=2,
            encoder_width=2,
            output_width=2,
            gamma=0.9,
            actor_lamda=0.8,
            critic_lamda=0.8,
            actor_alpha=0.01,
            critic_alpha=0.01,
            actor_kappa=1.0,
            critic_kappa=1.0,
            layer_norm_epsilon=1.0,
            normalize_observations=False,
            normalize_rewards=False,
            entropy_coefficient=0.0,
        )
    )


@pytest.mark.parametrize("incoming", [0.0, 0.25, 0.9])
@pytest.mark.parametrize("outgoing,boundary", [(0.1, False), (0.0, True), (0.7, True)])
def test_delayed_reward_uses_incoming_discount(
    agent: RecurrentTraceActorCriticAgent, incoming: float, outgoing: float, boundary: bool
) -> None:
    observation = jnp.zeros(2, dtype=jnp.float32)
    initial, first_action, first_policy = agent.start(agent.init(2, jr.key(230)), observation)
    np.testing.assert_array_equal(first_policy, [0.5, 0.5])
    first = agent.update(
        initial,
        jnp.float32(0.0),
        observation,
        discount=jnp.float32(incoming),
        episode_boundary=jnp.asarray(False),
    )
    assert bool(first.update_applied)
    np.testing.assert_array_equal(first.policy, [0.5, 0.5])
    second = agent.update(
        first.state,
        jnp.float32(1.0),
        observation,
        discount=jnp.float32(outgoing),
        episode_boundary=jnp.asarray(boundary),
    )
    assert bool(second.update_applied)
    assert float(second.td_error) == 1.0
    assert float(second.actor_obgd_scale) == float(second.critic_obgd_scale) == 1.0

    # Public initialization plus zero observations gives zero values and
    # uniform policies. The first reward is zero, so this second reward is
    # the only parameter update. The head-bias derivatives are analytic.
    first_gradient = np.eye(2)[int(first_action)] - 0.5
    second_gradient = np.eye(2)[int(first.action)] - 0.5
    expected_actor = agent.config.actor_alpha * (
        second_gradient + incoming * agent.config.actor_lamda * first_gradient
    )
    expected_critic = agent.config.critic_alpha * (1 + incoming * agent.config.critic_lamda)
    np.testing.assert_allclose(
        second.state.actor_params.head_bias, expected_actor, rtol=2e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        second.state.critic_params.head_bias, [expected_critic], rtol=2e-6, atol=1e-8
    )
    if boundary:
        for leaf in jax.tree.leaves((second.state.actor_traces, second.state.critic_traces)):
            np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_rejected_transition_preserves_discount_for_retry(
    agent: RecurrentTraceActorCriticAgent,
) -> None:
    observation = jnp.zeros(2, dtype=jnp.float32)
    initial, _, _ = agent.start(agent.init(2, jr.key(231)), observation)
    first = agent.update(initial, jnp.float32(0.0), observation, discount=jnp.float32(0.25))
    assert bool(first.update_applied)
    rejected = agent.update(first.state, jnp.float32(3e38), observation, discount=jnp.float32(0.7))
    assert not bool(rejected.update_applied)
    for actual, expected in zip(
        jax.tree.leaves(rejected.state), jax.tree.leaves(first.state), strict=True
    ):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = jr.key_data(actual), jr.key_data(expected)
        np.testing.assert_array_equal(actual, expected)

    # A same-runtime round trip must retain the discount as well as weights.
    restored = pickle.loads(pickle.dumps(rejected.state))
    retried = agent.update(restored, jnp.float32(1.0), observation, terminated=jnp.asarray(True))
    assert bool(retried.update_applied)
    np.testing.assert_allclose(retried.state.critic_params.head_bias, [0.012], rtol=2e-6)


def test_explicit_start_discards_previous_episode_credit(
    agent: RecurrentTraceActorCriticAgent,
) -> None:
    observation = jnp.zeros(2, dtype=jnp.float32)
    initial, _, _ = agent.start(agent.init(2, jr.key(232)), observation)
    first = agent.update(initial, jnp.float32(0.0), observation, discount=jnp.float32(0.25))
    restarted, _, _ = agent.start(first.state, observation)
    result = agent.update(restarted, jnp.float32(1.0), observation, terminated=jnp.asarray(True))
    assert bool(result.update_applied)
    np.testing.assert_allclose(result.state.critic_params.head_bias, [0.01], rtol=2e-6)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_legacy_positional_state_uses_configured_constant_discount(
    agent: RecurrentTraceActorCriticAgent, enable_x64: bool
) -> None:
    observation = jnp.zeros(2, dtype=jnp.float32)
    initial, _, _ = agent.start(agent.init(2, jr.key(233)), observation)
    first = agent.update(initial, jnp.float32(0.0), observation)
    # The original full positional prefix includes both optional moment trees.
    legacy = type(first.state)(*first.state[:-1])
    assert legacy.previous_discount is None
    expected = agent.update(
        first.state, jnp.float32(1.0), observation, terminated=jnp.asarray(True)
    )
    with jax.enable_x64(enable_x64):
        actual = agent.update(legacy, jnp.float32(1.0), observation, terminated=jnp.asarray(True))
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
            actual_leaf, expected_leaf = jr.key_data(actual_leaf), jr.key_data(expected_leaf)
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_invalid_saved_discount_cannot_apply_an_update(
    agent: RecurrentTraceActorCriticAgent,
) -> None:
    observation = jnp.zeros(2, dtype=jnp.float32)
    state, _, _ = agent.start(agent.init(2, jr.key(234)), observation)

    @jax.jit
    def compiled(discount: jax.Array, next_observation: jax.Array) -> jax.Array:
        return agent.update_from_started_state(
            state.replace(previous_discount=discount), jnp.float32(1.0), next_observation
        ).update_applied

    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        discount = jnp.float32(invalid)
        with pytest.raises(ValueError, match="state.previous_discount"):
            agent.update(state.replace(previous_discount=discount), jnp.float32(1.0), observation)
        assert not bool(compiled(discount, observation))
    assert bool(compiled(jnp.float32(0.25), observation))
