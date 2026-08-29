"""Reject deep runtime-profile mappings before json.dumps RecursionError.

Origin ``validate_environment_runtime_profile`` clones the caller mapping
with ``json.dumps`` and no nesting preflight. A 16_000-deep object nest
RecursionError's the C encoder on origin/main. Overlay fail-closes at the
shared 32-deep JSON ceiling before dumps.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import pytest
from test_runtime_profile import _matched_gpu_profile

from alberta_framework.benchmarks.runtime_profile import (
    _JSON_MAX_DEPTH,
    _JSON_MAX_NODES,
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


class _CountingListSubclass(list[int]):
    def __init__(self, values: list[int]) -> None:
        super().__init__(values)
        self.iterated = 0

    def __iter__(self) -> Iterator[int]:
        for value in super().__iter__():
            self.iterated += 1
            yield value


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


def test_json_list_subclass_respects_node_limit() -> None:
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy(_ListSubclass([0] * 5000), label="subclass-list")


def test_json_subclass_stops_without_eagerly_copying_all_children() -> None:
    payload = _CountingListSubclass([0] * (_JSON_MAX_NODES * 3))
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy(payload, label="counting-subclass-list")
    assert payload.iterated <= _JSON_MAX_NODES


def test_json_dict_subclass_respects_node_limit() -> None:
    big = _DictSubclass({str(i): i for i in range(5000)})
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy(big, label="subclass-dict")


def test_json_tuple_subclass_respects_node_limit() -> None:
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy(_TupleSubclass(range(5000)), label="subclass-tuple")


def test_json_list_subclass_nested_in_mapping_respects_node_limit() -> None:
    with pytest.raises(ValueError, match="resource limit"):
        _json_copy({"key": _ListSubclass([0] * 5000)}, label="nested-subclass")


def test_json_str_still_treated_as_leaf_not_container() -> None:
    result = _json_copy({"s": "hello"}, label="str-leaf")
    assert result == {"s": "hello"}
