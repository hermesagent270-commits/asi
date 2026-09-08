"""Ordinary temperature policies retain the established float32 evaluation order."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.actor_critic import ActorCriticAgent, ActorCriticConfig


@pytest.mark.parametrize("temperature", [0.001, 0.01, 0.25, 0.3, 0.7, 0.99, 1.0, 3.0])
@pytest.mark.parametrize("n_actions", [2, 5])
def test_finite_scaled_policy_retains_direct_softmax_bits(
    temperature: float, n_actions: int
) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=n_actions, temperature=temperature))
    state = agent.init(1, jr.key(190))
    observations = jnp.zeros((1,), dtype=jnp.float32)
    scales = jnp.tile(jnp.array([1e-3, 1.0, 10.0, 1e3], dtype=jnp.float32), 4)
    logits = jr.normal(jr.key(191), (16, n_actions)) * scales[:, None]
    assert bool(jnp.all(jnp.isfinite(logits / temperature)))

    # This is main's policy expression, compiled in the same batched context.
    direct_policy = jax.jit(jax.vmap(lambda row: jax.nn.softmax(row / temperature)))
    candidate_policy = jax.jit(
        jax.vmap(lambda row: agent.policy(state.replace(actor_bias=row), observations))
    )
    np.testing.assert_array_equal(candidate_policy(logits), direct_policy(logits))


@pytest.mark.parametrize("bad_logit", [jnp.nan, jnp.inf, -jnp.inf])
def test_cooling_fallback_preserves_nonfinite_input_rejection(bad_logit: float) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2, temperature=0.7))
    observation = jnp.ones((1,), dtype=jnp.float32)
    state = agent.init(1, jr.key(192)).replace(
        actor_bias=jnp.array([bad_logit, 1.0], dtype=jnp.float32),
        last_observation=observation,
    )
    result = agent.update(state, jnp.float32(1.0), observation)
    assert not bool(result.update_applied)
    assert int(result.state.step_count) == 0
    np.testing.assert_array_equal(result.state.actor_bias, state.actor_bias)
    np.testing.assert_array_equal(result.state.critic_weights, state.critic_weights)


@pytest.mark.parametrize("temperature", [0.25, 2.0**-127])
@pytest.mark.parametrize("batched", [False, True])
def test_temperature_scaling_preserves_policy_derivatives(
    batched: bool, temperature: float
) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2, temperature=temperature))
    state = agent.init(1, jr.key(196))
    observation = jnp.ones((1,), dtype=jnp.float32)

    def policy(bias: jax.Array) -> jax.Array:
        return agent.policy(state.replace(actor_bias=bias), observation)

    biases = jnp.array([[2e38, 2e38], [-2e38, -2e38], [0.0, 0.0]], dtype=jnp.float32)
    expected = jnp.array([[1.0, -1.0], [-1.0, 1.0]], dtype=jnp.float32) * (0.25 / temperature)
    if batched:
        # Batched fallback selection must not expose inactive NaN arithmetic
        # to reverse-mode differentiation.
        actual = jax.jacrev(jax.vmap(policy))(biases)
        reference = jnp.einsum("ij,ab->iajb", jnp.eye(len(biases)), expected)
        np.testing.assert_array_equal(actual, reference)
    else:
        for bias in biases:
            np.testing.assert_array_equal(jax.jacrev(policy)(bias), expected)


@pytest.mark.parametrize("temperature", [2.0**-127, 2.0**-149])
def test_cooling_fallback_keeps_ties_at_subnormal_temperature(temperature: float) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2, temperature=temperature))
    state = agent.init(1, jr.key(197))
    for value in (0.0, 2e38, -2e38):
        tied = state.replace(actor_bias=jnp.full((2,), value, dtype=jnp.float32))
        np.testing.assert_array_equal(agent.policy(tied, jnp.ones(1)), [0.5, 0.5])


def test_heating_preserves_main_policy_derivative_at_zero() -> None:
    temperature = 3.0
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2, temperature=temperature))
    state = agent.init(1, jr.key(198))
    observation = jnp.ones((1,), dtype=jnp.float32)

    def policy(bias: jax.Array) -> jax.Array:
        return agent.policy(state.replace(actor_bias=bias), observation)

    zero = jnp.zeros((2,), dtype=jnp.float32)

    def direct(bias: jax.Array) -> jax.Array:
        return jax.nn.softmax(bias / temperature)

    np.testing.assert_array_equal(jax.jacrev(policy)(zero), jax.jacrev(direct)(zero))
