"""Strict evidence validators must fail closed on JSON integers beyond float range.

``json.loads`` admits arbitrarily large integers, and ``float(10**400)`` raises
``OverflowError`` rather than ``ValueError``.  Each validator's numeric gate
must therefore report such a value as non-finite instead of crashing the
validator (and the ``--verify`` CLIs, which only map ``ValueError`` to exit 2).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from alberta_framework.evaluation import (
    continual_ia_artifact,
    continual_multiagent_artifact,
    ftl_decision_artifact,
    recurring_feature_artifact,
    scale_robust_feature_artifact,
)

pytestmark = pytest.mark.unit

_GATES: tuple[tuple[str, Callable[[object], float | None]], ...] = (
    ("continual_ia", continual_ia_artifact._number),
    ("continual_multiagent", continual_multiagent_artifact._finite_number),
    ("ftl_decision", ftl_decision_artifact._finite_number),
    ("recurring_feature", recurring_feature_artifact._finite_number),
    ("scale_robust_feature", scale_robust_feature_artifact._finite_number),
)


@pytest.mark.parametrize(("name", "gate"), _GATES, ids=[name for name, _ in _GATES])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_overflowing_integer_is_rejected_not_raised(
    name: str, gate: Callable[[object], float | None], value: int
) -> None:
    assert gate(value) is None, name


@pytest.mark.parametrize(("name", "gate"), _GATES, ids=[name for name, _ in _GATES])
def test_largest_float_representable_integer_is_still_accepted(
    name: str, gate: Callable[[object], float | None]
) -> None:
    value = int(1.7976931348623157e308)
    assert gate(value) == 1.7976931348623157e308, name
    assert gate(3) == 3.0, name
