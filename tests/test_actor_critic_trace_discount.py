"""Actor and critic delayed credit uses the preceding transition's discount."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("outgoing_discount", [0.0, 0.7])
@pytest.mark.parametrize("disable_jit", [False, True])
@pytest.mark.parametrize("fixed_actions", [False, True])
def test_array_runner_credits_previous_discount(
    continuous: bool, outgoing_discount: float, disable_jit: bool, fixed_actions: bool
) -> None:
    # The first transition has no reward, so both parameter vectors stay zero.
    # At the second reward, the first feature's eligibility is 0.4 * 0.8,
    # independently of the second transition's bootstrap discount or cfg.gamma.
    kwargs = dict(
        gamma=0.9,
        actor_lamda=0.8,
        critic_lamda=0.8,
        actor_step_size=0.1,
        critic_step_size=0.1,
    )
    observations = jnp.eye(2, dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)
    rewards = jnp.array([0.0, 1.0], dtype=jnp.float32)
    discounts = jnp.array([0.4, outgoing_discount], dtype=jnp.float32)
    with jax.disable_jit(disable_jit):
        if continuous:
            agent = ContinuousActorCriticAgent(
                ContinuousActorCriticConfig(action_dim=1, log_sigma_init=0.0, **kwargs)
            )
            result = run_continuous_actor_critic_from_arrays(
                agent,
                agent.init(2, jr.key(170)),
                observations,
                rewards,
                None,
                next_observations,
                actions=jnp.array([[2.0], [1.0]], dtype=jnp.float32) if fixed_actions else None,
                discounts=discounts,
            )
            actor_weights = result.state.mean_weights
            sampled = np.asarray(result.actions, dtype=np.float64)[:, 0]
            expected_actor = [[0.032 * sampled[0], 0.1 * sampled[1]]]
            np.testing.assert_allclose(
                result.state.log_sigma,
                [0.032 * (sampled[0] ** 2 - 1.0) + 0.1 * (sampled[1] ** 2 - 1.0)],
                rtol=1e-6,
                atol=1e-7,
            )
            actor_trace = result.state.mean_trace_weights
        else:
            agent = ActorCriticAgent(ActorCriticConfig(n_actions=2, **kwargs))
            result = run_actor_critic_from_arrays(
                agent,
                agent.init(2, jr.key(170)),
                observations,
                rewards,
                None,
                next_observations,
                actions=jnp.zeros((2,), dtype=jnp.int32) if fixed_actions else None,
                discounts=discounts,
            )
            actor_weights = result.state.actor_weights
            signs = 1.0 - 2.0 * np.asarray(result.actions)
            expected_actor = [
                [0.016 * signs[0], 0.05 * signs[1]],
                [-0.016 * signs[0], -0.05 * signs[1]],
            ]
            actor_trace = result.state.actor_trace_weights
    np.testing.assert_array_equal(result.updates_applied, [True, True])
    np.testing.assert_allclose(result.state.critic_weights, [0.032, 0.1], rtol=1e-6)
    np.testing.assert_allclose(result.state.critic_bias, 0.132, rtol=1e-6)
    np.testing.assert_allclose(actor_weights, expected_actor, rtol=1e-6)
    if outgoing_discount == 0.0:
        np.testing.assert_array_equal(result.state.critic_trace_weights, [0.0, 0.0])
        np.testing.assert_array_equal(actor_trace, jnp.zeros_like(actor_trace))


@pytest.fixture(params=[False, True], ids=["discrete", "continuous"])
def agent(request: pytest.FixtureRequest) -> ActorCriticAgent | ContinuousActorCriticAgent:
    kwargs = dict(
        gamma=0.9,
        actor_lamda=0.8,
        critic_lamda=0.8,
        actor_step_size=0.1,
        critic_step_size=0.1,
    )
    if request.param:
        return ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1, **kwargs))
    return ActorCriticAgent(ActorCriticConfig(n_actions=2, **kwargs))


def test_rejected_transition_preserves_discount_and_delayed_credit(agent) -> None:
    state = agent.init(3, jr.key(171))
    observations = jnp.eye(3, dtype=jnp.float32)
    state = agent.start(state, observations[0])[0]
    first = agent.update(state, jnp.float32(0.0), observations[1], discount=jnp.float32(0.4))
    assert bool(first.update_applied)
    assert float(first.state.previous_discount) == pytest.approx(0.4)
    rejected = agent.update(
        first.state, jnp.float32(jnp.nan), observations[2], discount=jnp.float32(0.7)
    )
    assert not bool(rejected.update_applied)
    for old, new in zip(jax.tree.leaves(first.state), jax.tree.leaves(rejected.state), strict=True):
        if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            old, new = jr.key_data(old), jr.key_data(new)
        np.testing.assert_array_equal(old, new)
    terminal = agent.update(
        rejected.state, jnp.float32(1.0), observations[2], discount=jnp.float32(0.0)
    )
    assert bool(terminal.update_applied)
    np.testing.assert_allclose(terminal.state.critic_weights, [0.032, 0.1, 0.0], rtol=1e-6)
    assert float(terminal.state.previous_discount) == 0.0
    restarted = agent.start(terminal.state, observations[2])[0]
    after_reset = agent.update(
        restarted, jnp.float32(1.0), observations[2], discount=jnp.float32(0.9)
    )
    assert bool(after_reset.update_applied)
    np.testing.assert_array_equal(
        after_reset.state.critic_weights[:2], terminal.state.critic_weights[:2]
    )
    assert float(after_reset.state.previous_discount) == pytest.approx(0.9)


def test_legacy_terminal_flag_credits_prior_trace(agent) -> None:
    state = agent.init(2, jr.key(174))
    observations = jnp.eye(2, dtype=jnp.float32)
    state = agent.start(state, observations[0])[0]
    first = agent.update(state, jnp.float32(0.0), observations[1], terminated=jnp.bool_(False))
    terminal = agent.update(
        first.state, jnp.float32(1.0), jnp.zeros((2,)), terminated=jnp.bool_(True)
    )
    assert bool(first.update_applied) and bool(terminal.update_applied)
    np.testing.assert_allclose(terminal.state.critic_weights, [0.072, 0.1], rtol=1e-6)
    np.testing.assert_array_equal(terminal.state.critic_trace_weights, [0.0, 0.0])


def test_previous_discount_counts_toward_persistent_bytes(agent) -> None:
    from alberta_framework.core.actor_critic import (
        _actor_critic_persistent_bytes,
        _continuous_actor_critic_persistent_bytes,
    )

    state = agent.init(3, jr.key(175))
    physical_leaves = [
        jr.key_data(leaf) if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf
        for leaf in jax.tree.leaves(state)
    ]
    actual_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in physical_leaves)
    if isinstance(agent, ActorCriticAgent):
        assert actual_bytes == _actor_critic_persistent_bytes(2, 3)
    else:
        assert actual_bytes == _continuous_actor_critic_persistent_bytes(1, 3)


@pytest.mark.parametrize("previous_discount", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_previous_discount_rejects_update(agent, previous_discount: float) -> None:
    observation = jnp.ones((1,), dtype=jnp.float32)
    state = agent.start(agent.init(1, jr.key(172)), observation)[0].replace(
        previous_discount=jnp.float32(previous_discount)
    )
    result = agent.update(state, jnp.float32(1.0), observation, discount=jnp.float32(0.5))
    assert not bool(result.update_applied)
    assert int(result.state.step_count) == 0
    np.testing.assert_array_equal(result.state.previous_discount, state.previous_discount)
    np.testing.assert_array_equal(result.state.critic_weights, state.critic_weights)


@pytest.mark.parametrize("previous_discount", [jnp.ones((1,)), jnp.int32(1)])
def test_previous_discount_requires_scalar_float32(agent, previous_discount: jax.Array) -> None:
    state = agent.init(1, jr.key(173)).replace(previous_discount=previous_discount)
    with pytest.raises(ValueError, match="previous_discount must be a scalar float32"):
        agent.update(state, jnp.float32(1.0), jnp.ones((1,)), discount=jnp.float32(0.5))
