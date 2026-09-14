"""Forager roots must not depend on process-wide JAX PRNG configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.forager import (
    ForagerBenchmarkConfig,
    ForagerEnvConfig,
    _agent_key,
    _recurrent_key,
    _rtu_rtrl_key,
    forager_rng_contract,
    run_forager,
)


class _KeyRecordingEnvironment:
    default_params = None

    def __init__(self) -> None:
        self.implementations: list[str] = []

    def reset(self, key: Any, params: Any) -> tuple[Any, Any]:
        del params
        self.implementations.append(str(jr.key_impl(key)))
        return jnp.zeros((1,), dtype=jnp.float32), jnp.asarray(0, dtype=jnp.int32)

    def step(
        self,
        key: Any,
        state: Any,
        action: Any,
        params: Any,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        del action, params
        self.implementations.append(str(jr.key_impl(key)))
        return (
            jnp.zeros((1,), dtype=jnp.float32),
            state + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False),
            {"biome_regret": jnp.asarray(0.0, dtype=jnp.float32)},
        )


class _ConstantPolicy:
    name = "constant"
    privileged = False

    def start(self, observation: Any, context: Any = None) -> int:
        del observation, context
        return 0

    def step(self, reward: float, observation: Any, context: Any = None) -> int:
        del reward, observation, context
        return 0

    def metadata(self) -> Mapping[str, Any]:
        return {"name": self.name, "privileged": self.privileged}


def test_forager_agent_roots_ignore_ambient_prng_implementation() -> None:
    root_functions = (_agent_key, _recurrent_key, _rtu_rtrl_key)
    expected_words = [np.asarray(jr.key_data(function(7))) for function in root_functions]

    with jax.default_prng_impl("rbg"):
        actual = [function(7) for function in root_functions]
        batched = [
            jax.jit(jax.vmap(function))(jnp.arange(4, dtype=jnp.uint32))
            for function in root_functions
        ]

    assert [str(jr.key_impl(key)) for key in actual] == ["threefry2x32"] * 3
    assert [str(jr.key_impl(keys)) for keys in batched] == ["threefry2x32"] * 3
    assert [jr.key_data(keys).shape for keys in batched] == [(4, 2)] * 3
    for key, expected in zip(actual, expected_words, strict=True):
        np.testing.assert_array_equal(jr.key_data(key), expected)


def test_forager_rng_contract_publishes_pinned_implementation() -> None:
    contract = forager_rng_contract()

    assert contract["schema_version"] == "alberta.forager_rng_schedule.v2"
    assert contract["prng_implementation"] == "threefry2x32"


@pytest.mark.integration
def test_host_runner_uses_threefry_environment_keys_under_ambient_rbg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _KeyRecordingEnvironment()
    monkeypatch.setattr(ForagerEnvConfig, "make", lambda self: (environment, None))
    config = ForagerBenchmarkConfig(steps=2, record_every=1, final_window=1)

    with jax.default_prng_impl("rbg"):
        result = run_forager(_ConstantPolicy(), config)

    assert result.steps == 2
    assert environment.implementations == ["threefry2x32"] * 3
