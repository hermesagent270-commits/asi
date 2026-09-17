"""Reconcile retained TeLAPA smoke records and reexecute their strict validator.

This consumes only public, already-consumed smoke roots. It is permanently
nonpromoting and does not execute a prospective campaign or the paper method.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks import telapa_qualification as telapa

SCHEMA = "asi.telapa_review_reconciliation.v1"
REPLAY = "development_result_main_f3d32c45_py3123_replay.v2.json"
HISTORICAL = (
    "development_result_pr2257.v2.json",
    "development_result_pr2257_review.v2.json",
)
ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "review_reconciliation.v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load(path: Path) -> Any:
    data = path.read_bytes()
    if len(data) > 1_000_000:
        raise ValueError("review input exceeds 1 MB")
    value = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    canonical(value)  # Also rejects overflowed JSON exponents.
    return value


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in record.items() if key != "resource_receipt"},
        **{f"resource_receipt.{key}": value for key, value in record["resource_receipt"].items()},
    }


def build_report() -> dict[str, Any]:
    expected_source = ROOT.parents[1] / "alberta_framework/benchmarks/telapa_qualification.py"
    if Path(telapa.__file__).resolve() != expected_source:
        raise RuntimeError("run from the repository root with PYTHONPATH=.")
    if jax.config.jax_default_prng_impl != "threefry2x32":
        raise ValueError("historical smoke replay requires the Threefry ambient default")
    replay = load(ROOT / REPLAY)
    # The existing validator independently reconstructs every bound record.
    # No identity field is replaced to make a historical result pass.
    telapa.validate_result(replay)
    records = replay["records"]
    current_pairs = {(row["seed"], row["arm"]): row for row in records}
    if len(current_pairs) != 12:
        raise ValueError("expected twelve unique current records")
    bindings = []
    comparisons = []
    rejection_records = []
    for name in (REPLAY, *HISTORICAL):
        value = load(ROOT / name)
        bindings.append(
            {
                "path": name,
                "file_sha256": sha((ROOT / name).read_bytes()),
                "records_sha256": sha(canonical(value["records"])),
                "record_count": len(value["records"]),
            }
        )
        if name == REPLAY:
            continue
        historical_pairs = {(row["seed"], row["arm"]): row for row in value["records"]}
        if (
            len(historical_pairs) != len(value["records"])
            or historical_pairs.keys() != current_pairs.keys()
        ):
            raise ValueError("historical/current arm-seed roster differs")
        # Equal records do not imply equal source/runtime provenance. Keep all
        # historical identity differences rather than normalizing them away.
        identity_matches = {
            field: canonical(value["identity"][field]) == canonical(replay["identity"][field])
            for field in replay["identity"]
        }
        identity_differences = {
            field: {"historical": value["identity"][field], "replay": replay["identity"][field]}
            for field, matches in identity_matches.items()
            if not matches
        }
        for field in replay:
            if field not in ("identity", "records") and canonical(value[field]) != canonical(
                replay[field]
            ):
                raise ValueError(f"historical root field differs: {field}")
        for pair, row in sorted(current_pairs.items()):
            old = flatten(historical_pairs[pair])
            new = flatten(row)
            if old.keys() != new.keys():
                raise ValueError("record axes differ")
            axis_matches = {
                field: canonical(old[field]) == canonical(new[field]) for field in sorted(new)
            }
            if not all(axis_matches.values()):
                raise ValueError(f"historical record differs: {name} {pair}")
            comparisons.append(
                {
                    "historical_path": name,
                    "seed": pair[0],
                    "arm": pair[1],
                    "historical_record_sha256": sha(canonical(historical_pairs[pair])),
                    "replay_record_sha256": sha(canonical(row)),
                    "axis_matches": axis_matches,
                }
            )
        try:
            telapa.validate_result(value)
        except ValueError as error:
            expected = "result identity differs from the current source/runtime/registries"
            if str(error) != expected or not identity_differences:
                raise
            rejection_records.append(
                {
                    "path": name,
                    "accepted": False,
                    "error": str(error),
                    "identity_field_matches": identity_matches,
                    "identity_differences": identity_differences,
                }
            )
        else:
            raise ValueError("historical artifact unexpectedly accepted")

    adapter = telapa.SwitchingPolicyLifeAdapter(phase_length=4, learning_rate=0.125)
    seed_initialization = []
    for seed in replay["config"]["seeds"]:
        state, policy = adapter.init(jr.key(seed, impl="threefry2x32"))
        policy_sha = sha(telapa._policy_bytes(policy))
        if any(
            row["initial_policy_sha256"] != policy_sha for row in records if row["seed"] == seed
        ):
            raise ValueError("seed initialization differs from retained policy")
        seed_initialization.append(
            {
                "seed": seed,
                "initial_state_index": int(state.state_index),
                "initial_greedy_actions_by_state": np.argmax(policy, axis=1).tolist(),
                "initial_policy_sha256": policy_sha,
            }
        )
    coincidences = []
    for arm in sorted({row["arm"] for row in records}):
        selected = [row for row in records if row["arm"] == arm]
        for field in (
            "initial_policy_sha256",
            "final_policy_sha256",
            "action_sha256",
            "observation_sha256",
            "reward_sha256",
        ):
            groups: dict[str, list[int]] = {}
            for row in selected:
                groups.setdefault(row[field], []).append(row["seed"])
            coincidences.append(
                {
                    "arm": arm,
                    "axis": field,
                    "equal_hash_groups": [
                        {"sha256": digest, "seeds": sorted(seeds)}
                        for digest, seeds in sorted(groups.items())
                        if len(seeds) > 1
                    ],
                }
            )
    return {
        "schema": SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "artifact_bindings": bindings,
        "canonicalization": "UTF-8 json.dumps(sort_keys=True,separators=(',',':'),allow_nan=False)",
        "record_comparisons": comparisons,
        "seed_initialization": seed_initialization,
        "cross_seed_coincidences": coincidences,
        "validator": {
            "command": (
                "PYTHONPATH=. /root/asi/.venv/bin/python "
                "outputs/telapa_qualification/review_audit.v1.py --check"
            ),
            "validator_schema": telapa.SCHEMA,
            "validator_source_sha256": sha(Path(telapa.__file__).read_bytes()),
            "audit_source_sha256": sha(Path(__file__).read_bytes()),
            "validated_identity": replay["identity"],
            "jax_default_prng_impl": jax.config.jax_default_prng_impl,
            "jax_enable_x64": jax.config.jax_enable_x64,
            "exit_status": 0,
            "records_reexecuted": 12,
            "historical_rejections": rejection_records,
        },
        "attestation": (
            "consistency and observed local validation; no authenticated execution proof"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, nargs="?", const=REPORT)
    args = parser.parse_args()
    if args.check is None and REPORT.exists():
        raise FileExistsError(REPORT)
    report = build_report()
    if args.check is not None:
        if canonical(load(args.check)) != canonical(report):
            raise ValueError("retained review report differs from current audit")
    else:
        with REPORT.open("x", encoding="utf-8") as output:
            json.dump(report, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
    print(
        "Accepted: 12 strict record reexecutions; "
        "24 historical/current comparisons; all axes match."
    )


if __name__ == "__main__":
    main()
