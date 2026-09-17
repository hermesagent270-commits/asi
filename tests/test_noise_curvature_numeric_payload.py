"""Bind the scheduler's byte receipt to its admitted numeric payload."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    NoiseCurvatureConfig,
    init_noise_curvature_state,
    noise_curvature_persistent_bytes,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params
from alberta_framework.core.baseline_optimizers import Adam

pytestmark = pytest.mark.unit


def params():
    config = IPMNISTConfig(
        n_tasks=1, task_length=40, input_dim=4, hidden1=3, hidden2=2, n_classes=2
    )
    values = init_mlp_params(jax.random.key(3, impl="threefry2x32"), config)
    # The real parameter creator explicitly allocates float32 under both x64
    # defaults; pin that fact rather than hiding a promoted dtype with a cast.
    assert all(value.dtype == np.dtype(np.float32) for value in values.values())
    return config, values


@pytest.mark.parametrize(
    "container,dtype,x64,only",
    [
        ("numpy", "float64", True, None),
        ("jax", "float64", True, None),
        ("numpy", "float16", False, None),
        ("jax", "float16", False, None),
        ("jax", "bfloat16", False, None),
        ("numpy", "float64", True, "b3"),
        ("jax", "float64", True, "b3"),
    ],
)
def test_noncanonical_dtype_rejected_before_adam_allocation(
    monkeypatch, container, dtype, x64, only
):
    with jax.enable_x64(x64):
        _, values = params()
        convert = np.asarray if container == "numpy" else jnp.asarray
        values = {
            name: convert(value, dtype=dtype) if only is None or name == only else value
            for name, value in values.items()
        }
        allocations = []
        original = Adam.init_for_shape

        def observed(self, shape):
            allocations.append(shape)
            return original(self, shape)

        monkeypatch.setattr(Adam, "init_for_shape", observed)
        with pytest.raises(ValueError, match="float32"):
            init_noise_curvature_state(
                values, NoiseCurvatureConfig(mode="combined", total_steps=40)
            )
        assert not allocations


@pytest.mark.parametrize("container", ["numpy", "jax"])
@pytest.mark.parametrize("x64", [False, True])
def test_admitted_float32_payload_matches_actual_leaf_bytes(container, x64):
    with jax.enable_x64(x64):
        config, values = params()
        if container == "numpy":
            values = {name: np.asarray(value) for name, value in values.items()}
        state = init_noise_curvature_state(
            values, NoiseCurvatureConfig(mode="combined", total_steps=40)
        )
        actual = sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves((values, state)))
        assert actual == noise_curvature_persistent_bytes(
            parameter_count=config.parameter_count,
            input_dim=config.input_dim,
            control_interval=40,
        )


def test_numpy_float64_conversion_remains_supported_with_x64_disabled():
    with jax.enable_x64(False):
        config, values = params()
        host_values = {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}
        state = init_noise_curvature_state(
            host_values, NoiseCurvatureConfig(mode="combined", total_steps=40)
        )
        canonical = {name: jnp.asarray(value) for name, value in host_values.items()}
        assert all(value.dtype == np.dtype(np.float32) for value in canonical.values())
        actual = sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves((canonical, state)))
        assert actual == noise_curvature_persistent_bytes(
            parameter_count=config.parameter_count,
            input_dim=config.input_dim,
            control_interval=40,
        )
