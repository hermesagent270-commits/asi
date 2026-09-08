"""Terminal critic credit stays inside the episode that produced its eligibility."""

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import DemonType, GVFSpec, HordeLearner, create_horde_spec
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    run_horde_actor_critic_from_arrays,
)
from alberta_framework.core.types import TraceMode

pytestmark = pytest.mark.integration


def _critic(trace_mode=TraceMode.ACCUMULATING, *, gamma0=False):
    demons = [
        GVFSpec(
            name=f"head{index}",
            demon_type=DemonType.PREDICTION,
            gamma=0.0 if gamma0 and index == 0 else 0.9,
            lamda=0.8,
            cumulant_index=index,
        )
        for index in range(2)
    ]
    return HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(),
        step_size=0.1,
        sparsity=0.0,
        trace_mode=trace_mode,
    )


def _zero_params(state):
    params = state.head_params
    return state.replace(
        head_params=params.replace(
            weights=tuple(jnp.zeros_like(x) for x in params.weights),
            biases=tuple(jnp.zeros_like(x) for x in params.biases),
        ),
        birth_timestamp=0.0,
        uptime_s=0.0,
    )


@pytest.mark.parametrize("value_index", [0, 1])
@pytest.mark.parametrize("trace_mode", [TraceMode.ACCUMULATING, TraceMode.REPLACING])
def test_public_runner_does_not_credit_a_finished_episode(value_index, trace_mode):
    critic = _critic(trace_mode)
    agent = HordeActorCriticAgent(
        HordeActorCriticConfig(n_actions=1, actor_lamda=0.0, value_head_index=value_index),
        critic,
    )
    state = agent.init(2, jr.key(200))
    state = state.replace(critic_state=_zero_params(state.critic_state))
    result = run_horde_actor_critic_from_arrays(
        agent,
        state,
        observations=jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        rewards=jnp.array([0.0, 1.0, 1.0]),
        next_observations=jnp.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        actions=jnp.zeros(3, dtype=jnp.int32),
        auxiliary_cumulants=jnp.array([[0.0], [0.0], [1.0]]),
        discounts=jnp.array([0.9, 0.0, 0.0]),
    )
    assert bool(jnp.all(result.updates_applied))
    # Episode one's terminal reward credits both visited features. Episode
    # two has zero features, so its reward cannot change those weights.
    np.testing.assert_allclose(
        result.state.critic_state.head_params.weights[value_index],
        [[0.072, 0.1]],
        rtol=1e-6,
        atol=1e-8,
    )
    for trace in result.state.critic_state.head_traces[value_index]:
        np.testing.assert_array_equal(trace, jnp.zeros_like(trace))
    # The auxiliary head keeps its continuing eligibility across this boundary.
    aux_index = 1 - value_index
    np.testing.assert_allclose(
        result.state.critic_state.head_traces[aux_index][0],
        [[0.5184, 0.72]],
        rtol=1e-6,
        atol=1e-8,
    )
    assert bool(jnp.any(result.state.critic_state.head_params.weights[aux_index] != 0))


@pytest.mark.parametrize("failure", ["inactive", "cumulant", "discount", "observation"])
def test_unaccepted_terminal_head_keeps_its_eligibility(failure):
    critic = _critic()
    state = _zero_params(critic.init(2, jr.key(201)))
    warm = critic.update(state, jnp.array([1.0, 0.0]), jnp.zeros(2), jnp.zeros(2)).state
    cumulants = jnp.array([1.0, 1.0])
    discounts = jnp.array([0.0, 0.9])
    observation = jnp.array([0.0, 1.0])
    if failure == "inactive":
        cumulants = cumulants.at[0].set(jnp.nan)
    elif failure == "cumulant":
        cumulants = cumulants.at[0].set(jnp.inf)
    elif failure == "discount":
        discounts = discounts.at[0].set(jnp.nan)
    else:
        observation = observation.at[0].set(jnp.nan)
    result = critic.update_with_discounts(warm, observation, cumulants, jnp.zeros(2), discounts)
    assert not bool(result.head_updates_applied[0])
    chex.assert_trees_all_equal(result.state.head_traces[0], warm.head_traces[0])
    chex.assert_trees_all_equal(result.state.head_params.weights[0], warm.head_params.weights[0])
    if failure == "observation":
        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, warm)
    else:
        assert bool(result.update_applied)
        assert bool(result.head_updates_applied[1])


@pytest.mark.parametrize("gamma0", [False, True])
def test_default_discounts_preserve_the_existing_update_exactly(gamma0):
    critic = _critic(gamma0=gamma0)
    state = _zero_params(critic.init(2, jr.key(202)))
    observation = jnp.array([1.0, 0.0])
    next_observation = jnp.array([0.0, 1.0])
    for cumulants in (jnp.array([0.5, 1.0]), jnp.array([1.0, -0.5])):
        ordinary = critic.update(state, observation, cumulants, next_observation)
        explicit = critic.update_with_discounts(
            state, observation, cumulants, next_observation, critic.horde_spec.gammas
        )
        chex.assert_trees_all_equal(ordinary, explicit)
        state = ordinary.state
