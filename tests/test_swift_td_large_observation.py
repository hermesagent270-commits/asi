"""Finite large features must not turn SwiftTD's bound into a zero update."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.learners import TDLinearLearner
from alberta_framework.core.swift_td import SwiftTD


@pytest.mark.parametrize("scale", [1e10, 1e19, 1e20, 1e30])
@pytest.mark.parametrize("decay", [0.99, 1.0])
@pytest.mark.parametrize("alpha", [0.01, 1e-30])
def test_public_learner_preserves_representable_bounded_update(scale, decay, alpha):
    optimizer = SwiftTD(
        initial_step_size=alpha,
        eta=0.1,
        eta_min=1e-35,
        meta_step_size=0.0,
        step_size_decay=decay,
    )
    learner = TDLinearLearner(optimizer)
    state = learner.init(4)
    obs = jnp.array([scale, -0.5 * scale, 0.0, 0.25 * scale], dtype=jnp.float32)
    result = jax.jit(learner.update)(state, obs, jnp.array(1.0), jnp.zeros(4), jnp.array(0.0))
    assert bool(result.update_applied)
    phi = np.append(np.asarray(obs, dtype=np.float64), 1.0)
    alphas = np.exp(np.asarray(state.optimizer_state.log_step_sizes, dtype=np.float64))
    tau = np.sum(alphas * phi**2)
    expected = min(1.0, float(optimizer.to_config()["eta"]) / tau) * alphas * phi
    actual = np.append(np.asarray(result.state.weights), float(result.state.bias))
    # Only require nonzero storage where the mathematical result is normal float32.
    representable = np.abs(expected) >= np.finfo(np.float32).tiny
    np.testing.assert_allclose(actual[representable], expected[representable], rtol=3e-5, atol=0)
    expected_prediction = min(tau, 0.1)
    prediction = float(jnp.squeeze(learner.predict(result.state, obs)))
    assert prediction == pytest.approx(expected_prediction, rel=3e-5, abs=1e-15)


def test_bound_respects_heterogeneous_step_sizes_and_signed_features():
    optimizer = SwiftTD(initial_step_size=0.01, eta=0.1, eta_min=1e-35, meta_step_size=0.0)
    state = optimizer.init(4).replace(
        log_step_sizes=jnp.log(jnp.array([1e-30, 0.01, 1e-10, 0.1, 1e-3]))
    )
    obs = jnp.array([1e30, -1e20, 0.0, 1e10], dtype=jnp.float32)
    result = jax.jit(optimizer.update)(state, jnp.array(1.0), obs, jnp.zeros(4), jnp.array(0.0))
    assert bool(result.update_applied)
    phi = np.append(np.asarray(obs, np.float64), 1.0)
    alpha = np.exp(np.asarray(state.log_step_sizes, np.float64))
    expected = float(state.eta) * alpha * phi / np.sum(alpha * phi**2)
    actual = np.append(np.asarray(result.weight_delta), float(result.bias_delta))
    normal = np.abs(expected) >= np.finfo(np.float32).tiny
    np.testing.assert_allclose(actual[normal], expected[normal], rtol=3e-5, atol=0)
    assert float(result.weight_delta[2]) == 0.0


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_stable_bound_still_rejects_nonfinite_observations_atomically(bad):
    optimizer = SwiftTD(initial_step_size=0.01)
    state = optimizer.init(2)
    result = jax.jit(optimizer.update)(
        state, jnp.array(1.0), jnp.array([1e20, bad]), jnp.zeros(2), jnp.array(0.0)
    )
    assert not bool(result.update_applied)
    for before, after in zip(
        jax.tree.leaves(state), jax.tree.leaves(result.new_state), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(result.weight_delta, [0.0, 0.0])


@pytest.mark.parametrize("decay", [0.99, 1.0])
def test_repeated_public_updates_learn_from_large_observation(decay):
    learner = TDLinearLearner(
        SwiftTD(
            initial_step_size=0.01,
            eta=0.1,
            eta_min=1e-35,
            meta_step_size=0.0,
            trace_decay=0.0,
            step_size_decay=decay,
        )
    )
    state = learner.init(1)
    obs = jnp.array([1e20])
    update = jax.jit(learner.update)
    for step in range(1, 9):
        result = update(state, obs, jnp.array(1.0), jnp.zeros(1), jnp.array(0.0))
        assert bool(result.update_applied)
        state = result.state
        assert float(jnp.squeeze(learner.predict(state, obs))) == pytest.approx(
            1.0 - 0.9**step, rel=3e-5
        )
