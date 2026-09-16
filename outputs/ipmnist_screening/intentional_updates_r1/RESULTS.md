# Supervised Intentional Updates development result

This append-only development screen completes the supervised measurement slice
of issue #1561. It is permanently nonpromoting and makes no publication-parity,
scientific, `reference-dev`, whole-agent, or robotics claim.

## Decision

**Rejected.** The full supervised Intentional Updates arm averaged
`0.760557 ± 0.000882` mean online accuracy (mean ± sample standard deviation,
five development seeds), while its exact mechanism-off control averaged
`0.791919 ± 0.001805`. The paired delta was
`-0.031362 ± 0.001296` (standard error `0.000580`), and every paired seed was
negative:

```text
-0.030427, -0.032617, -0.032900, -0.030733, -0.030133
```

This fails the frozen requirement of a paired mean greater than `+0.005` with
all five seeds positive and meets the preregistered rejection condition of a
nonpositive mean. No arm is promoted.

## Arm results

Mean online accuracy across 60 tasks, reported as mean ± sample standard
deviation over seeds `1561000`–`1561004`:

| Arm | Accuracy | Paired delta vs off | Persistent numeric bytes |
|---|---:|---:|---:|
| `intentional_updates_off` | 0.791919 ± 0.001805 | — | 1,134,916 |
| `intentional_updates_no_diag` | 0.768355 ± 0.000733 | -0.023564 | 2,263,568 |
| `intentional_updates_no_clip` | 0.760582 ± 0.000943 | -0.031337 | 2,263,568 |
| `intentional_updates_ipmnist` | 0.760557 ± 0.000882 | -0.031362 | 2,263,568 |
| `intentional_updates_head_only` | 0.632189 ± 0.002460 | -0.159731 | 2,263,568 |

Removing diagonal normalization reduced the loss but did not rescue the
mechanism: it remained negative on every paired seed. Removing adaptive
clipping was practically identical to the full arm at this horizon. Freezing
feature updates was substantially worse. These ablation readings are
descriptive and do not change the primary rejection.

## Protocol and resources

- Exact measured source: `8c235f21ce8a7d7fe01c4c8bc712877dd0eaef3a`.
- Relevant-source SHA-256:
  `5c908bb81e548f355276825232fde23d2d089862ccaa7eedd7b77257fd4e0aea`.
- Plan SHA-256:
  `a420446877eebeb516c826e710b4f8aaeb2424cad2f33f0281fd1d05988b8d4a`.
- Audit SHA-256:
  `b7487e9ad65f22d2ff7a9777388d30e453dfd402e7c306d71ff81c3b798038b2`.
- Dataset: OpenML `mnist_784` v1 rows 0–59,999, materialized as float32
  inputs and int32 labels; every shard has one matching dataset identity.
- Runtime: Linux CPU, Python 3.12.3, JAX 0.9.1, `OMP_NUM_THREADS=1`; every
  shard has one matching runtime identity.
- Each arm/seed consumed 300,000 observations, 300,000 updates, 300,000
  backward passes, and 600,000 model queries. Timing includes compilation and
  is telemetry only.
- Total timing telemetry was 320.05 seconds for mechanism-off, 3,217.99 for
  no-diagonal, 4,642.72 for no-clipping, 4,612.52 for full Intentional, and
  3,360.69 for head-only. These totals are not selection metrics.

The audit strictly reloads all 25 shards, reconstructs and validates every
Intentional Updates resource receipt, requires the complete five-arm by
five-seed Cartesian product, recomputes the generic summary, and binds shard,
plan, source, dataset, runtime, and audit-script digests.

## Commands

The preregistered shards ran from the measured commit with four workers:

```bash
xargs -n 2 -P 4 outputs/ipmnist_screening/intentional_updates_r1/worker.sh \
  < outputs/ipmnist_screening/intentional_updates_r1/jobs.txt
```

The aggregate and independent audit ran with the recorded environment:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m \
  alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/ipmnist_screening/intentional_updates_r1/shards/*.json \
  --control-name intentional_updates_off --slope-window 15 \
  --output outputs/ipmnist_screening/intentional_updates_r1/summary.json

OMP_NUM_THREADS=1 .venv/bin/python \
  outputs/ipmnist_screening/intentional_updates_r1/audit.py \
  outputs/ipmnist_screening/intentional_updates_r1/audit.json
```

Before execution, a broken local cache symlink caused all 25 worker invocations
to fail during dataset-home creation. No model initialized and no shard was
published, so no seed was consumed by that preflight incident. The 25 logs are
retained under `preflight_failures/`. The cache was then materialized and
validated once before the successful campaign. A first summary derivation also
failed closed before publication because it omitted the recorded
`OMP_NUM_THREADS=1`; the successful command above supplied the exact binding.
