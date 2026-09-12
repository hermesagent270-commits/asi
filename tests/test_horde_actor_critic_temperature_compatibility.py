"""Temperature-scaling regressions shared by all Horde actor forms."""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec


def _critic(*, control: bool) -> HordeLearner:
    demons = [
        GVFSpec(  # type: ignore[call-arg]
            name=f"head_{index}",
            demon_type=DemonType.CONTROL if control else DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=-1,
        )
        for index in range(2 if control else 1)
    ]
    return HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(),
        step_size=0.01,
        use_layer_norm=False,
    )


def _linear_agents(temperature: float) -> tuple[Any, ...]:
    return (
        HordeActorCriticAgent(
            HordeActorCriticConfig(
                n_actions=2,
                actor_step_size=0.05,
                actor_lamda=0.7,
                temperature=temperature,
            ),
            _critic(control=False),
        ),
        QHordeActorCriticAgent(
            QHordeActorCriticConfig(
                n_actions=2,
                gamma=0.9,
                actor_step_size=0.05,
                actor_lamda=0.7,
                temperature=temperature,
            ),
            _critic(control=True),
        ),
    )


def _nonlinear_agents(temperature: float) -> tuple[Any, ...]:
    return (
        NonlinearHordeActorCriticAgent(
            NonlinearHordeActorCriticConfig(
                n_actions=2,
                hidden_sizes=(),
                actor_sparsity=0.0,
                temperature=temperature,
            ),
            _critic(control=False),
        ),
        NonlinearQHordeActorCriticAgent(
            NonlinearQHordeActorCriticConfig(
                n_actions=2,
                hidden_sizes=(),
                actor_sparsity=0.0,
                temperature=temperature,
            ),
            _critic(control=True),
        ),
    )


@pytest.mark.parametrize("agent_index", [0, 1])
def test_finite_linear_update_keeps_non_power_of_two_cooled_policy_finite(
    agent_index: int,
) -> None:
    """An accepted finite update must not make the next public policy unusable."""
    agent = _linear_agents(0.7)[agent_index]
    observation = jnp.ones((1,), dtype=jnp.float32)
    state = agent.init(feature_dim=1, key=jr.key(10 + agent_index)).replace(
        last_observation=observation,
        last_action=jnp.asarray(0, dtype=jnp.int32),
    )

    update_kwargs = (
        {"terminated": jnp.asarray(False)} if isinstance(agent, QHordeActorCriticAgent) else {}
    )
    result = agent.update(
        state,
        reward=jnp.asarray(3e38, dtype=jnp.float32),
        observation=observation,
        **update_kwargs,
    )
    assert bool(result.update_applied)
    chex.assert_tree_all_finite(result.state.replace(rng_key=jr.key_data(result.state.rng_key)))

    next_observation = jnp.asarray([30.0], dtype=jnp.float32)
    logits = result.state.actor_weights @ next_observation + result.state.actor_bias
    chex.assert_tree_all_finite(logits)
    probabilities = agent.policy(result.state, next_observation)

    chex.assert_tree_all_finite(probabilities)
    np.testing.assert_allclose(np.asarray(probabilities), np.asarray([1.0, 0.0]))
    action, _next_key, sampled_probabilities = agent.select_action(result.state, next_observation)
    assert int(action) == 0
    chex.assert_tree_all_finite(sampled_probabilities)


@pytest.mark.parametrize("agent_index", [0, 1])
def test_nonlinear_policy_and_update_keep_finite_extreme_logits_usable(
    agent_index: int,
) -> None:
    """The nonlinear policy and its differentiated update share stable scaling."""
    agent = _nonlinear_agents(0.7)[agent_index]
    observation = jnp.ones((1,), dtype=jnp.float32)
    state = agent.init(feature_dim=1, key=jr.key(20 + agent_index)).replace(
        actor_head_b=jnp.asarray([3e38, 2e38], dtype=jnp.float32),
        last_observation=observation,
        last_action=jnp.asarray(0, dtype=jnp.int32),
    )

    probabilities = agent.policy(state, observation)
    chex.assert_tree_all_finite(probabilities)
    np.testing.assert_allclose(np.asarray(probabilities), np.asarray([1.0, 0.0]))
    action, _next_key, sampled_probabilities = agent.select_action(state, observation)
    assert int(action) == 0
    chex.assert_tree_all_finite(sampled_probabilities)

    update_kwargs = (
        {"terminated": jnp.asarray(0.0, dtype=jnp.float32)}
        if isinstance(agent, NonlinearQHordeActorCriticAgent)
        else {}
    )
    result = agent.update(
        state,
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        observation=observation,
        **update_kwargs,
    )
    assert bool(result.update_applied)
    chex.assert_tree_all_finite(result.policy)


@pytest.mark.parametrize(
    "agent_factory",
    [
        pytest.param(lambda temperature: _linear_agents(temperature)[0], id="linear"),
        pytest.param(lambda temperature: _nonlinear_agents(temperature)[0], id="nonlinear"),
    ],
)
def test_large_temperature_preserves_opposite_finite_logit_separation(
    agent_factory: Callable[[float], Any],
) -> None:
    """Pre-scaling centering must not overflow a finite heated distribution."""
    agent = agent_factory(1e38)
    observation = jnp.ones((1,), dtype=jnp.float32)
    state = agent.init(feature_dim=1, key=jax.random.key(30))
    if isinstance(agent, HordeActorCriticAgent):
        state = state.replace(actor_bias=jnp.asarray([3e38, -3e38], dtype=jnp.float32))
    else:
        state = state.replace(actor_head_b=jnp.asarray([3e38, -3e38], dtype=jnp.float32))

    probabilities = agent.policy(state, observation)

    chex.assert_tree_all_finite(probabilities)
    np.testing.assert_allclose(
        np.asarray(probabilities),
        np.asarray(jax.nn.softmax(jnp.asarray([3.0, -3.0], dtype=jnp.float32))),
        rtol=1e-5,
    )
