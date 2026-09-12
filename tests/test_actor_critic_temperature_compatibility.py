"""Temperature scaling stays finite without changing exact-scale policies."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.actor_critic import ActorCriticAgent, ActorCriticConfig


@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("n_actions", [2, 5])
def test_power_of_two_temperature_retains_direct_softmax_bits(
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


@pytest.mark.parametrize(
    ("logits", "temperature"),
    [
        ([2e9, 0.0], 0.7),
        ([1e20, 1e20 * (1.0 + 2.0**-23)], 0.3),
        ([1e38, 1.0, -1e38], 0.99),
        ([1.0, -2.0, 0.5], 0.7),
    ],
)
def test_non_power_of_two_cooling_matches_eager_policy(
    logits: list[float], temperature: float
) -> None:
    """Unbatched compilation must preserve the finite eager policy."""
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=len(logits), temperature=temperature))
    state = agent.init(1, jr.key(199)).replace(actor_bias=jnp.asarray(logits, dtype=jnp.float32))
    observation = jnp.zeros(1, dtype=jnp.float32)
    with jax.disable_jit(True):
        expected = jax.nn.softmax(state.actor_bias / temperature)

    actual = agent.policy(state, observation)
    assert bool(jnp.all(jnp.isfinite(actual)))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.0)


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


def test_cooling_overflow_reached_by_learning_keeps_next_transition() -> None:
    agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            temperature=0.25,
            actor_step_size=0.05,
            actor_lamda=0.0,
            critic_lamda=0.0,
        )
    )
    observation = jnp.ones(1, dtype=jnp.float32)
    state, action, _ = agent.start(agent.init(1, jr.key(220)), observation)
    first = agent.update(state, jnp.float32(3e38), observation)
    assert bool(first.update_applied)
    assert int(first.state.step_count) == 1

    # The finite reward learns large weights from zero initialization. The
    # next feature changes from 1 to 2 after the policy has saturated; zero
    # policy gradients do not prevent this larger logit at a new observation.
    next_observation = 2.0 * observation
    logits = first.state.actor_weights @ next_observation + first.state.actor_bias
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert not bool(jnp.all(jnp.isfinite(logits / agent.config.temperature)))
    second = agent.update(first.state, jnp.float32(0.0), next_observation)
    assert bool(second.update_applied)
    assert int(second.state.step_count) == 2
    for leaf in jax.tree.leaves(second.state):
        if not jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            assert bool(jnp.all(jnp.isfinite(leaf)))
    np.testing.assert_array_equal(
        agent.policy(second.state, next_observation), jax.nn.one_hot(action, 2)
    )


@pytest.mark.parametrize("seed", range(5))
def test_non_power_of_two_cooling_reached_by_learning_remains_finite(seed: int) -> None:
    """Compiled policy evaluation must not turn a learned finite policy into NaN."""
    agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            temperature=0.7,
            actor_step_size=0.1,
            critic_step_size=0.1,
            actor_lamda=0.0,
            critic_lamda=0.0,
        )
    )
    observation = jnp.array([1.5e5], dtype=jnp.float32)
    zero_observation = jnp.zeros(1, dtype=jnp.float32)
    state, _, _ = agent.start(agent.init(1, jr.key(seed)), observation)
    first = agent.update(state, jnp.float32(1.0), zero_observation)
    assert bool(first.update_applied)

    logits = first.state.actor_weights @ observation + first.state.actor_bias
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert bool(jnp.all(jnp.isfinite(logits / agent.config.temperature)))

    policy = agent.policy(first.state, observation)
    assert bool(jnp.all(jnp.isfinite(policy)))
    assert int(jnp.argmax(policy)) == int(jnp.argmax(logits))
    second = agent.update(first.state, jnp.float32(0.0), observation)
    assert bool(second.update_applied)


@pytest.mark.parametrize("temperature", [0.25, 2.0**-127, 2.0**-149])
@pytest.mark.parametrize("disable_jit", [False, True])
def test_cooling_recovery_supports_nan_debugger(temperature: float, disable_jit: bool) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2, temperature=temperature))
    state = agent.init(1, jr.key(221))
    observation = jnp.ones(1, dtype=jnp.float32)
    for bias, expected in (
        ([1e38, 2e38], [0.0, 1.0]),
        ([2e38, 2e38], [0.5, 0.5]),
        ([0.0, 0.0], [0.5, 0.5]),
    ):
        configured = state.replace(actor_bias=jnp.array(bias, dtype=jnp.float32))
        with jax.disable_jit(disable_jit), jax.debug_nans(True):
            actual = agent.policy(configured, observation)
            actual.block_until_ready()
        np.testing.assert_array_equal(actual, expected)
