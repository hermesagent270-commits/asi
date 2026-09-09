"""Prototype behavior gradients must support the zero-option control surface."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.core.state_builder import OnlineGatedStateBuilderConfig

pytestmark = pytest.mark.unit


def _agent() -> PrototypeAgent:
    builder = OnlineGatedStateBuilderConfig(
        observation_dim=2,
        n_actions=1,
        hidden_dim=1,
        step_size=0.01,
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(),
                    observation_dim=builder.feature_dim(),
                    n_primitive_actions=1,
                    base_step_size=0.1,
                    base_avg_reward_step_size=0.0,
                    epsilon_base=0.0,
                )
            ),
            state_builder=builder,
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=builder.feature_dim(),
                mode="behavior_only",
            ),
        )
    )


def _transition(state: PrototypeAgentState) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        next_decision_observation=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
    )


def test_behavior_only_mixer_updates_without_an_option_bank() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(310, impl="threefry2x32")),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
    )

    result = agent.update_transition(state, _transition(state))

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_gradient_result.valid)
    assert bool(result.representation_gradient_mix.applied)
    assert bool(result.state_builder_learning_diagnostics.applied)
    assert int(result.state.state_builder_state.update_count) == 1
    assert np.isfinite(np.asarray(result.behavior_representation_gradient)).all()


def test_compiled_behavior_only_mixer_updates_without_an_option_bank() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(311, impl="threefry2x32")),
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
    )

    result = jax.jit(agent.update_transition)(state, _transition(state))

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_gradient_result.valid)
    assert bool(result.representation_gradient_mix.applied)
    assert bool(result.state_builder_learning_diagnostics.applied)
    assert int(result.state.state_builder_state.update_count) == 1
    assert np.isfinite(np.asarray(result.behavior_representation_gradient)).all()
