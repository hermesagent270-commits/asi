"""Integer SpaceSpec bounds and values never round through NumPy float64.

NumPy has no common integer dtype for builtin ints that straddle the int64 and
uint64 ranges (for example ``[2**63 + 1025, 1]``) and silently promotes them to
float64, which rounds every integer above 2**53.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.reference_agent import SpaceSpec

pytestmark = pytest.mark.unit

_DECLARED_HIGH = 2**63 + 1025


def _uint64_box() -> SpaceSpec:
    return SpaceSpec.box(
        shape=(2,),
        dtype="uint64",
        low=[0, 0],
        high=[_DECLARED_HIGH, 1],
        semantic_id="act",
    )


def test_uint64_box_keeps_declared_high_bound_exactly() -> None:
    spec = _uint64_box()

    assert spec.high == (_DECLARED_HIGH, 1)
    assert spec.low == (0, 0)


def test_uint64_box_rejects_value_above_the_declared_bound() -> None:
    spec = _uint64_box()
    beyond = np.array([_DECLARED_HIGH + 1, 0], dtype=np.uint64)

    with pytest.raises(ValueError, match="outside the declared bounds"):
        spec.encode(beyond)
    encoded = spec.encode(np.array([_DECLARED_HIGH, 1], dtype=np.uint64))
    assert encoded.to_python() == (_DECLARED_HIGH, 1)


def test_uint64_box_encodes_builtin_integers_across_the_int64_boundary() -> None:
    spec = SpaceSpec.box(
        shape=(2,),
        dtype="uint64",
        low=None,
        high=None,
        semantic_id="act",
    )

    encoded = spec.encode([2**64 - 1, 5])

    assert encoded.to_python() == (2**64 - 1, 5)


def test_integer_bounds_keep_existing_rejections() -> None:
    with pytest.raises(ValueError, match="integral values representable"):
        SpaceSpec.box(
            shape=(2,),
            dtype="uint64",
            low=[0, 0.5],
            high=[2**63, 1],
            semantic_id="act",
        )
    with pytest.raises(ValueError, match="not representable by uint64"):
        SpaceSpec.box(
            shape=(2,),
            dtype="uint64",
            low=[-1, 0],
            high=[2**63, 1],
            semantic_id="act",
        )
    spec = SpaceSpec.box(
        shape=(2,), dtype="uint64", low=None, high=None, semantic_id="act"
    )
    with pytest.raises(TypeError, match="not float"):
        spec.encode([2**63, 5.0])
