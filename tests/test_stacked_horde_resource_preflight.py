"""Resource-derived stacked-Horde preflights before sequence traversal."""

from __future__ import annotations

import pytest

from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    _decode_sequence,
    nexting_spec,
)

_INT32_MAX = 2**31 - 1


def test_config_rejects_oversized_demon_envelope_before_sequence_lengths() -> None:
    with pytest.raises(ValueError, match="stacked Horde aggregate"):
        StackedHordeConfig(
            n_demons=_INT32_MAX,
            feature_dim=1,
            gammas=(),
            lamdas=(),
            cumulant_indices=(),
        )


def test_config_rejects_oversized_resource_envelope_before_element_validation() -> None:
    with pytest.raises(ValueError, match="stacked Horde aggregate"):
        StackedHordeConfig(
            n_demons=1,
            feature_dim=_INT32_MAX,
            gammas=(object(),),  # type: ignore[arg-type]
            lamdas=(0.5,),
            cumulant_indices=(0,),
        )


def test_deserializer_preflights_resources_before_decoding_sequences() -> None:
    payload = {
        "type": "StackedHordeConfig",
        "n_demons": 1,
        "feature_dim": _INT32_MAX,
        "gammas": [object()],
        "lamdas": [0.5],
        "cumulant_indices": [0],
        "step_size": 0.05,
    }

    with pytest.raises(ValueError, match="stacked Horde aggregate"):
        StackedHordeConfig.from_config(payload)


def test_serialized_sequence_length_is_checked_before_tuple_copy() -> None:
    with pytest.raises(ValueError, match="gammas must have length n_demons=1"):
        _decode_sequence("gammas", [0.9, 0.8], expected_length=1)


def test_nexting_preflights_derived_resources_before_element_validation() -> None:
    with pytest.raises(ValueError, match="stacked Horde aggregate"):
        nexting_spec(
            _INT32_MAX,
            (object(),),  # type: ignore[arg-type]
            gammas=(0.5,),
        )


def test_resource_valid_4097_demon_config_and_deserialization_remain_supported() -> None:
    n_demons = 4097
    config = StackedHordeConfig(
        n_demons=n_demons,
        feature_dim=1,
        gammas=(0.9,) * n_demons,
        lamdas=(0.5,) * n_demons,
        cumulant_indices=(0,) * n_demons,
    )

    assert config.n_demons == n_demons
    assert StackedHordeConfig.from_config(config.to_config()) == config


def test_resource_valid_4225_demon_nexting_grid_remains_supported() -> None:
    config = nexting_spec(
        feature_dim=1,
        cumulant_indices=(0,) * 65,
        gammas=(0.5,) * 65,
    )

    assert config.n_demons == 4225
