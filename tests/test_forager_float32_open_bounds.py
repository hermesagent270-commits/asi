"""Open unit-interval bounds must hold in the float32 dtype the agents execute."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks.forager import (
    AlbertaForagerConfig,
    ForagerFeatureConfig,
    ForagerFeatureEncoder,
)

pytestmark = pytest.mark.unit

# Strictly below one in float64, but exactly 1.0 once cast to float32.
_ROUNDS_TO_ONE = 1.0 - 1e-9


def test_value_rounds_to_one_in_float32() -> None:
    assert _ROUNDS_TO_ONE < 1.0
    assert np.float32(_ROUNDS_TO_ONE) == np.float32(1.0)


def test_reward_trace_decay_rejects_float32_unit_endpoint() -> None:
    with pytest.raises(ValueError, match="reward_trace_decays"):
        ForagerFeatureConfig(reward_trace_decays=(_ROUNDS_TO_ONE,))


@pytest.mark.parametrize(
    "field_name",
    ["actor_epsilon", "td_error_normalizer_decay", "recurrent_scale"],
)
def test_alberta_open_unit_fields_reject_float32_unit_endpoint(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        AlbertaForagerConfig(**{field_name: _ROUNDS_TO_ONE})


@pytest.mark.parametrize(
    "overrides",
    [
        {"actor_epsilon": 0.0},
        {"td_error_normalizer_decay": 0.0},
        {"recurrent_scale": 0.0},
        {"actor_epsilon": float(np.nextafter(np.float32(1.0), np.float32(0.0)))},
        {"td_error_normalizer_decay": 0.9999999},
        {"recurrent_scale": 0.99},
    ],
)
def test_alberta_open_unit_fields_keep_representable_values(overrides: dict[str, Any]) -> None:
    AlbertaForagerConfig(**overrides)


def test_largest_float32_decay_below_one_still_updates_trace() -> None:
    decay = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
    encoder = ForagerFeatureEncoder(
        ForagerFeatureConfig(reward_trace_decays=(decay,), reward_scale=1.0)
    )
    state = encoder.advance(encoder.init(), action=0, reward=1.0)
    assert state.reward_traces[0] > 0.0
