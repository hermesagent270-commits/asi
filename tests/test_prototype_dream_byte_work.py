"""Configuration-derived byte-work bounds for Prototype dreaming."""

from __future__ import annotations

import pytest

from alberta_framework.core.oak import OaKConfig, _oak_update_working_set_bytes
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.core.world_model import (
    ActionConditionedWorldModelConfig,
    _world_model_direct_state_scalars,
)

_INT32_MAX = 2**31 - 1
_SCAN_WORD_BYTES = 8


def _oak(base_hidden_sizes: tuple[int, ...] = ()) -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=4,
            n_primitive_actions=2,
            base_hidden_sizes=base_hidden_sizes,
        )
    )


def _world_model(
    hidden_sizes: tuple[int, ...] = (),
    *,
    include_action_interactions: bool = False,
) -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=4,
        n_actions=2,
        hidden_sizes=hidden_sizes,
        include_action_interactions=include_action_interactions,
    )


def _per_dream_byte_work(
    oak: OaKConfig,
    world_model: ActionConditionedWorldModelConfig,
) -> int:
    action_feature_dim = world_model.n_actions
    if world_model.include_action_interactions:
        action_feature_dim += world_model.observation_dim * world_model.n_actions
    world_model_state_bytes = 4 * _world_model_direct_state_scalars(
        observation_dim=world_model.observation_dim,
        action_feature_dim=action_feature_dim,
        hidden_sizes=world_model.hidden_sizes,
        n_heads=world_model.observation_dim + 2,
        outer_state_scalars=2 * world_model.observation_dim + 4,
    )
    return (
        _oak_update_working_set_bytes(oak.stomp)
        + world_model_state_bytes
        + _SCAN_WORD_BYTES
    )


def test_int32_dream_count_is_rejected_by_configured_byte_work() -> None:
    with pytest.raises(ValueError, match="dream byte-work"):
        PrototypeAgentConfig(
            oak=_oak(),
            world_model=_world_model(),
            n_dreams_per_step=_INT32_MAX,
        )


def test_resource_valid_count_above_rejected_fixed_ceiling_remains_supported() -> None:
    config = PrototypeAgentConfig(
        oak=_oak(),
        world_model=_world_model(),
        n_dreams_per_step=10_001,
    )

    assert config.n_dreams_per_step == 10_001
    assert PrototypeAgentConfig.from_config(config.to_config()) == config


@pytest.mark.parametrize("hidden_sizes", [(), (64, 64)])
def test_direct_and_serialized_boundaries_follow_configured_byte_work(
    hidden_sizes: tuple[int, ...],
) -> None:
    oak = _oak()
    world_model = _world_model(hidden_sizes)
    per_dream = _per_dream_byte_work(oak, world_model)
    last_fit = _INT32_MAX // per_dream
    first_overflow = last_fit + 1

    config = PrototypeAgentConfig(
        oak=oak,
        world_model=world_model,
        n_dreams_per_step=last_fit,
    )
    assert config.n_dreams_per_step == last_fit

    with pytest.raises(ValueError, match="dream byte-work"):
        PrototypeAgentConfig(
            oak=oak,
            world_model=world_model,
            n_dreams_per_step=first_overflow,
        )

    payload = config.to_config()
    payload["n_dreams_per_step"] = first_overflow
    with pytest.raises(ValueError, match="dream byte-work"):
        PrototypeAgentConfig.from_config(payload)


def test_larger_configured_states_reduce_the_supported_dream_count() -> None:
    oak = _oak()
    linear_last_fit = _INT32_MAX // _per_dream_byte_work(oak, _world_model())
    mlp_last_fit = _INT32_MAX // _per_dream_byte_work(
        oak, _world_model((64, 64))
    )
    interaction_last_fit = _INT32_MAX // _per_dream_byte_work(
        oak,
        _world_model(include_action_interactions=True),
    )
    larger_oak_last_fit = _INT32_MAX // _per_dream_byte_work(
        _oak((64,)),
        _world_model(),
    )

    assert linear_last_fit > mlp_last_fit > 10_001
    assert linear_last_fit > interaction_last_fit > 10_001
    assert linear_last_fit > larger_oak_last_fit > 10_001
