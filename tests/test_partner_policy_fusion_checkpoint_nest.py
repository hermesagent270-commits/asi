"""Reject deep checkpoint mappings before json.dumps RecursionError.

Origin ``PartnerPolicyFusion.from_checkpoint_payload`` digest-binds the
caller ``fusion`` mapping with ``json.dumps`` and no nesting preflight.
A 16_000-deep object nest RecursionError's the C encoder on origin/main.
Overlay fail-closes at the shared 32-deep JSON ceiling before dumps.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from typing import cast

import pytest

from alberta_framework.core.partner_policy_fusion import (
    _CHECKPOINT_JSON_MAX_DEPTH,
    _CHECKPOINT_JSON_MAX_NODES,
    MECHANISM_STATUS,
    PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA,
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    _canonical_json_bytes,
    _json_container_children,
)

pytestmark = pytest.mark.unit


def _nest(depth: int) -> dict[str, object]:
    node: dict[str, object] = {"leaf": 1}
    for _ in range(depth):
        node = {"x": node}
    return node


def _hostile_checkpoint(fusion: object) -> dict[str, object]:
    return {
        "schema": PARTNER_POLICY_FUSION_CHECKPOINT_SCHEMA,
        "mechanism_status": MECHANISM_STATUS,
        "scientific_promotion_allowed": False,
        "fusion": fusion,
        "config_digest": "0" * 64,
        "resource_budget": {},
        "state": {},
        "state_digest": "0" * 64,
    }


def test_frozen_checkpoint_json_nest_bound() -> None:
    assert _CHECKPOINT_JSON_MAX_DEPTH == 32


def test_last_fit_checkpoint_still_roundtrips() -> None:
    fusion = PartnerPolicyFusion(
        PartnerPolicyFusionConfig(
            max_partners=3,
            context_dim=2,
            n_actions=4,
            max_abs_context=1.0,
        )
    )
    payload = fusion.checkpoint_payload(fusion.init())
    restored_fusion, restored_state = PartnerPolicyFusion.from_checkpoint_payload(payload)
    assert restored_fusion.to_config() == fusion.to_config()
    assert int(restored_state.decision_count) == 0


def test_last_fit_json_chain_still_encodes() -> None:
    encoded = _canonical_json_bytes(_nest(_CHECKPOINT_JSON_MAX_DEPTH - 1))
    assert encoded.startswith(b"{")


def test_origin_recursion_class_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dumps(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("json.dumps ran before the checkpoint nest gate")

    monkeypatch.setattr(json, "dumps", fail_dumps)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting depth"):
        PartnerPolicyFusion.from_checkpoint_payload(_hostile_checkpoint(_nest(16_000)))
    assert time.perf_counter() - started < 0.25


class _DictSubclass(dict[str, object]):
    pass


class _ListSubclass(list[int]):
    pass


class _TupleSubclass(tuple[int, ...]):
    pass


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
    raise AssertionError("json.dumps ran before the checkpoint container gate")


@pytest.mark.parametrize(
    "payload",
    [
        _ListSubclass([0] * (_CHECKPOINT_JSON_MAX_NODES + 1)),
        _DictSubclass({str(i): i for i in range(_CHECKPOINT_JSON_MAX_NODES + 1)}),
        _TupleSubclass(tuple([0] * (_CHECKPOINT_JSON_MAX_NODES + 1))),
        _HiddenValuesDict({str(i): i for i in range(_CHECKPOINT_JSON_MAX_NODES + 1)}),
        _HiddenItemsDict({"real": 1}),
        _HiddenIterList(),
        {"fusion": [(_ListSubclass([1]),)]},
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
    with pytest.raises(ValueError, match="container subclass"):
        _canonical_json_bytes(payload)


def test_checkpoint_subclass_fusion_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(json, "dumps", _fail_dumps)
    hidden = _HiddenValuesDict({str(i): i for i in range(_CHECKPOINT_JSON_MAX_NODES + 1)})
    with pytest.raises(ValueError, match="container subclass"):
        PartnerPolicyFusion.from_checkpoint_payload(_hostile_checkpoint(hidden))


def test_exact_tuple_rejected_over_limit() -> None:
    with pytest.raises(ValueError, match="resource"):
        _canonical_json_bytes(tuple(range(5000)))


def test_exact_list_rejected_over_limit() -> None:
    with pytest.raises(ValueError, match="resource"):
        _canonical_json_bytes([0] * 5000)


def test_exact_dict_rejected_over_limit() -> None:
    with pytest.raises(ValueError, match="resource"):
        _canonical_json_bytes({str(i): i for i in range(5000)})


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
