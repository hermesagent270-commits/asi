"""Configuration-derived byte-work bounds for Prototype dreaming."""

from __future__ import annotations

import pytest

from alberta_framework.core import prototype_agent as prototype_agent_module
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgentConfig,
    _prototype_dream_iteration_byte_charge,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

_INT32_MAX = 2**31 - 1


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


def test_int32_dream_count_is_rejected_by_configured_byte_work() -> None:
    with pytest.raises(ValueError, match="dream byte-work"):
        PrototypeAgentConfig(
            oak=_oak(),
            world_model=_world_model(),
            n_dreams_per_step=_INT32_MAX,
        )


def test_resource_valid_count_above_ten_thousand_remains_supported() -> None:
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
    per_dream = _prototype_dream_iteration_byte_charge(oak, world_model)
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


def _assert_configured_boundary(
    oak: OaKConfig,
    world_model: ActionConditionedWorldModelConfig,
) -> int:
    last_fit = _INT32_MAX // _prototype_dream_iteration_byte_charge(oak, world_model)
    PrototypeAgentConfig(
        oak=oak,
        world_model=world_model,
        n_dreams_per_step=last_fit,
    )
    with pytest.raises(ValueError, match="dream byte-work"):
        PrototypeAgentConfig(
            oak=oak,
            world_model=world_model,
            n_dreams_per_step=last_fit + 1,
        )
    return last_fit


def test_larger_configured_states_reduce_the_supported_dream_count() -> None:
    linear_last_fit = _assert_configured_boundary(_oak(), _world_model())
    mlp_last_fit = _assert_configured_boundary(_oak(), _world_model((64, 64)))
    interaction_last_fit = _assert_configured_boundary(
        _oak(), _world_model(include_action_interactions=True)
    )
    larger_oak_last_fit = _assert_configured_boundary(_oak((64,)), _world_model())

    assert linear_last_fit > mlp_last_fit > 10_001
    assert linear_last_fit > interaction_last_fit > 10_001
    assert linear_last_fit > larger_oak_last_fit > 10_001


def test_exact_signed_int32_byte_work_fit_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prototype_agent_module,
        "_prototype_dream_iteration_byte_charge",
        lambda oak, world_model: _INT32_MAX,
    )

    PrototypeAgentConfig(oak=_oak(), world_model=_world_model(), n_dreams_per_step=1)
    with pytest.raises(ValueError, match="dream byte-work"):
        PrototypeAgentConfig(oak=_oak(), world_model=_world_model(), n_dreams_per_step=2)
