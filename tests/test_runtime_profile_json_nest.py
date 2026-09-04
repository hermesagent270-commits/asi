"""Reject deep runtime-profile mappings before json.dumps RecursionError.

Origin ``validate_environment_runtime_profile`` clones the caller mapping
with ``json.dumps`` and no nesting preflight. A 16_000-deep object nest
RecursionError's the C encoder on origin/main. Overlay fail-closes at the
shared 32-deep JSON ceiling before dumps.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from typing import cast

import pytest
from test_runtime_profile import _matched_gpu_profile

from alberta_framework.benchmarks.runtime_profile import (
    _JSON_MAX_DEPTH,
    _JSON_MAX_NODES,
    _json_container_children,
    _json_copy,
    validate_environment_runtime_profile,
)

pytestmark = pytest.mark.unit


class _ListSubclass(list[int]):
    """Subclass of list to verify ABC-based container recognition."""


class _DictSubclass(dict[str, object]):
    """Subclass of dict to verify ABC-based container recognition."""


class _TupleSubclass(tuple[int, ...]):
    """Subclass of tuple to verify ABC-based container recognition."""


def _nest(depth: int) -> dict[str, object]:
    node: dict[str, object] = {"leaf": 1}
    for _ in range(depth):
        node = {"x": node}
    return node


def test_frozen_runtime_profile_json_nest_bound() -> None:
    assert _JSON_MAX_DEPTH == 32


def test_last_fit_runtime_profile_still_validates() -> None:
    profile = _matched_gpu_profile()
    validated = validate_environment_runtime_profile(profile)
    assert validated["schema_version"] == profile["schema_version"]


def test_last_fit_json_chain_still_encodes() -> None:
    copied = _json_copy(_nest(_JSON_MAX_DEPTH - 1), label="runtime profile nest")
    assert isinstance(copied, dict)


def test_origin_recursion_class_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dumps(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("json.dumps ran before the runtime-profile nest gate")

    monkeypatch.setattr(json, "dumps", fail_dumps)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting depth"):
        validate_environment_runtime_profile(_nest(16_000))
    assert time.perf_counter() - started < 0.25


class _HiddenValuesDict(dict[str, object]):
    """``json.dumps`` walks ``items()``; a ``values()`` reader sees nothing."""

    def values(self) -> Iterator[object]:  # type: ignore[override]
        return iter(())


class _HiddenItemsDict(dict[str, object]):
    """``json.dumps`` walks ``items()``, which yields content absent from storage."""

    def items(self) -> list[tuple[str, object]]:  # type: ignore[override]
        return [("x", _nest(16_000))]


class _HiddenIterList(list[object]):
    """``json.dumps`` walks ``__iter__``, which yields content absent from storage."""

    def __iter__(self) -> Iterator[object]:
        yield _nest(16_000)


def _fail_dumps(*_args: object, **_kwargs: object) -> str:
    raise AssertionError("json.dumps ran before the runtime-profile container gate")


@pytest.mark.parametrize(
    "payload",
    [
        _ListSubclass([0] * (_JSON_MAX_NODES + 1)),
        _DictSubclass({str(i): i for i in range(_JSON_MAX_NODES + 1)}),
        _TupleSubclass(tuple([0] * (_JSON_MAX_NODES + 1))),
        _HiddenValuesDict({str(i): i for i in range(_JSON_MAX_NODES + 1)}),
        _HiddenItemsDict({"real": 1}),
        _HiddenIterList(),
        {"key": [(_ListSubclass([1]),)]},
    ],
    ids=[
        "list-subclass",
        "dict-subclass",
        "tuple-subclass",
        "dict-hidden-values",
        "dict-hidden-items",
        "list-hidden-iter",
        "nested-subclass",
    ],
)
def test_json_container_subclass_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    monkeypatch.setattr(json, "dumps", _fail_dumps)
    with pytest.raises(ValueError, match="subclass-gate contains a JSON container subclass"):
        _json_copy(payload, label="subclass-gate")


def test_runtime_profile_subclass_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(json, "dumps", _fail_dumps)
    hidden = _HiddenValuesDict({str(i): i for i in range(_JSON_MAX_NODES + 1)})
    with pytest.raises(ValueError, match="container subclass"):
        validate_environment_runtime_profile(hidden)


def test_exact_containers_reject_by_node_limit() -> None:
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy([0] * (_JSON_MAX_NODES + 1), label="exact-list")
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy({str(i): i for i in range(_JSON_MAX_NODES + 1)}, label="exact-dict")
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy(tuple([0] * (_JSON_MAX_NODES + 1)), label="exact-tuple")


def test_exact_container_children_are_lazy_views() -> None:
    exact_list = [1, 2]
    assert _json_container_children(exact_list) is exact_list
    exact_tuple = (1, 2)
    assert _json_container_children(exact_tuple) is exact_tuple
    exact_dict = {"a": 1}
    view = _json_container_children(exact_dict)
    assert type(view) is type({}.values())
    assert list(cast(Iterable[object], view)) == [1]
    for leaf in ("text", b"bytes", 1, 1.5, None, True):
        assert _json_container_children(leaf) is None


def test_json_str_still_treated_as_leaf_not_container() -> None:
    result = _json_copy({"s": "hello"}, label="str-leaf")
    assert result == {"s": "hello"}
