# Attempt history and review reconciliation

This supplement addresses the review of PR #2948 at
`c0c73d549136e38a38277b172e04355577b5ffd1`. The existing
[RESULTS.md](RESULTS.md), final paragraph, already records the preflight
incident and cache repair. [review_manifest.v1.json](review_manifest.v1.json)
adds the missing file-by-file bindings and dispositions. All 82 original
campaign files remain byte-identical to that reviewed commit. No consumed
campaign seed was rerun, no outcome changed, and no arm was promoted.

## Two attempts, one retained result

| Phase | Logged UTC interval | Disposition | Artifacts used in the aggregate |
|---|---|---|---|
| Dataset-home preflight | 2026-09-16 03:49:36.491–03:50:00.388 | All 25 invocations failed before the learner runner | None |
| Completed campaign | 2026-09-16 03:51:36.382–05:01:58.255 | All 25 planned arm/seed pairs completed | The 25 files in `shards/` |

Every `preflight_failures/` log ends with the same `FileExistsError` during
`load_mnist_train` → `fetch_openml` → dataset-home creation. Removing only the
initial log timestamp makes all 25 logs byte-identical; the normalized traceback
SHA-256 is `1aefaa7db4912163f6a44457cf9af98f72faf1d23383d2c8110d2fa1afe6c84d`.
At measured source `8c235f21ce8a7d7fe01c4c8bc712877dd0eaef3a`, the CLI loads
the dataset before calling `run_screening_config`, which initializes the model
and consumes its learner RNG. The tracebacks never reach that call. These
dataset-home failures supplied no learned trajectory and consumed no learner
seed; the successful campaign subsequently consumed the entire roster, which
must not be reused.

The cache materialization was an untracked local data repair, as reported in
the original report. There is no separately retained cache-repair transcript.
The successful shards all bind the same clean measured Git identity, package
source/lock digest, canonical dataset identity, and observed runtime identity.
Tracked package/config/lock bytes are unchanged between the measured commit
and the reviewed result commit. This reconciles the records' provenance; it
does not independently attest the repair or authenticate execution.

The manifest maps each `(arm, seed)` to its failed preflight log, successful
log, and published shard, with byte counts and SHA-256 digests. Each successful
log contains the matching arm/seed start, task 20/40/60 progress, and final
published shard path. Its four-decimal accuracy matches the retained curve
mean. Completion timestamps agree with the shard creation epochs within
0.01 seconds under a UTC interpretation. Timing is telemetry: logs format
the raw duration to one decimal, while shards round it to two decimals, so
double rounding can disagree at half-tenth boundaries. All differences fit
the formatting error bound of `0.05 + 0.005` seconds.

The 25 failed logs are retained deliberately to preserve the complete initial
attempt roster, as the preregistration requires. The 25 successful logs have
25 distinct hashes and pin different progress, timing, and output paths.
The repository requires append-only campaign artifacts; the supplement and
manifest resolve the duplication concern without deleting any historical log.

## Verified bindings and configuration

The retained auditor was run again as a read-only arithmetic audit and exactly
reconstructed `audit.json`. It strictly reloads the 25 shards, validates their
resource receipts and complete roster, recomputes the summary, and applies the
original primary rule. The decision remains `development_rejection` with
paired mean `-0.03136200136666665` and all five paired deltas negative.

| Retained file | Verified SHA-256 |
|---|---|
| `PREREGISTRATION.md` | `a420446877eebeb516c826e710b4f8aaeb2424cad2f33f0281fd1d05988b8d4a` |
| `summary.json` | `35048f91058b9c76a58b247f0e8086f8d0fd9874a90a66f0243c6ca7feec4828` |
| `audit.json` | `b7487e9ad65f22d2ff7a9777388d30e453dfd402e7c306d71ff81c3b798038b2` |
| `audit.py` | `047bd9ff32cf4f8aa6918e966d3a9c36279d244a3b1abc6df5c023dc05c5d554` |

The plan first appears at Git commit
`471f5c1e92d8c8555d14b58abeed9fdfa8f59cc2`, whose recorded commit time is
2026-09-16 03:48:30 UTC, preceding every retained attempt log. Its bytes are
identical at the measured source commit and in this bundle, and the original
audit already binds that plan hash. Git history and log timestamps provide
consistency evidence, not an independently authenticated preregistration time.

All five arms serialize the same 13 hyperparameter keys. Both `beta_clip` and
`clip_mult` are present in every no-diagonal shard. In the no-clipping arm they
are retained but inactive because `use_adaptive_clip` is exactly `0.0`;
`use_diagonal_normalization` is exactly `0.0` for the no-diagonal arm. The
strict retained-shard auditor accepted all arm configurations against the
existing registry. The manifest records the complete key set.

## Executable checks and discharge criterion

This is a retained-result PR. Its implementation coverage already lives in
`tests/test_intentional_updates_ipmnist.py`: mechanism-off versus fixed-step
JIT equality, head-only feature freezing, frozen hyperparameter rejection,
matched resource counters, and strict receipt policy/serialization checks.
The full existing file passed again: **16 tests**. These are tiny synthetic
implementation checks, not campaign reexecution or performance evidence.

The rejection does not remove the reusable registered arm or impose a new
execution authorization gate. The original consumed roster must not be rerun;
a future experiment requires a new causal hypothesis and protocol. This PR
does not claim to prevent someone from configuring such a future experiment.

The review findings are discharged when the attempt manifest covers the
complete roster, its hashes match the retained bytes, and the existing auditor
reconstructs the original audit exactly. The following read-only check performs
those hash and arithmetic checks from the repository root with the project
venv; it neither loads MNIST nor calls a campaign learner:

```bash
OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path("outputs/ipmnist_screening/intentional_updates_r1")
manifest = json.loads((root / "review_manifest.v1.json").read_text())
assert len(manifest["original_files"]) == 82
assert len(manifest["rows"]) == 25
bindings = {item["path"]: item for item in manifest["original_files"]}
for item in manifest["original_files"]:
    data = (root / item["path"]).read_bytes()
    assert len(data) == item["bytes"]
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
for row in manifest["rows"]:
    stem = f'{row["arm"]}_seed{row["seed"]}'
    for section, path in (
        ("preflight", f"preflight_failures/{stem}.log"),
        ("completed", f"logs/{stem}.log"),
        ("shard", f"shards/{stem}.json"),
    ):
        assert row[section]["path"] == path
        for key in ("sha256", "bytes"):
            assert row[section][key] == bindings[path][key]
assert {(row["arm"], row["seed"]) for row in manifest["rows"]} == {
    (arm, seed)
    for arm in json.loads((root / "audit.json").read_text())["plan"]["arms"]
    for seed in range(1561000, 1561005)
}
spec = importlib.util.spec_from_file_location("retained_audit", root / "audit.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
assert auditor.build_audit() == json.loads((root / "audit.json").read_text())
print("82 original file bindings and 25 attempt rows accepted; audit exactly matched")
PY
```

This is a consistency and arithmetic check. No dataset-bound learner replay,
independent execution attestation, scientific promotion, or reference-channel
change is supplied by this supplement.
