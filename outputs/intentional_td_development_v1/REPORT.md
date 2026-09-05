# Intentional TD: negative development comparison

All 40 preregistered runs completed successfully at `e540cc8ee562bb0987bb31531b4eebe914ca2f5c`: four development seeds × five arms × two environments, 10,000 transitions and updates per run. The full Intentional arm failed the preregistered benefit rule. No parameter or seed was changed after observing results.

Preregistration: https://github.com/SlopDotCash/asi/issues/1561#issuecomment-5550454893

Command (from the measured checkout; project venv):

```bash
taskset -c 0,1,2,3 /root/asi/.venv/bin/python -m alberta_framework.benchmarks.intentional_td_development \
  --output outputs/intentional_td_development_v1/results_e540cc8e.json \
  > outputs/intentional_td_development_v1/run_e540cc8e.log 2>&1
```

## Results

Reward per transition, mean ± sample standard deviation over seeds 15610, 15611, 15612, 15613. Every arm and seed is retained in the result JSON.

| Arm | SwitchingTwoState | RiverSwim (3 states) |
|---|---:|---:|
| intentional | 0.548425 ± 0.021051 | 0.243633 ± 0.319697 |
| intentional_no_trace | 0.725050 ± 0.010011 | 0.199402 ± 0.324727 |
| intentional_no_rms | 0.583400 ± 0.007161 | 0.442761 ± 0.293601 |
| fixed_trace | 0.715525 ± 0.006325 | 0.541589 ± 0.215576 |
| fixed_step | 0.493550 ± 0.014842 | 0.175385 ± 0.336882 |

The full arm loses to `fixed_trace` by 0.167100 mean reward on SwitchingTwoState (all four paired seeds negative) and 0.297956 on RiverSwim (two negative, two positive). RiverSwim seed spread is large. The no-trace arm performs better than the full arm on SwitchingTwoState, but was not the preregistered primary candidate and is not promoted. This rejects the stated benefit criterion for this linear configuration; it does not refute Intentional Updates generally, the official deep-RL implementation, or the supervised IPMNIST extension.

## Accounting and verification

- 400,000 environment transitions and optimizer updates; 1,200,040 logical prediction calls.
- Sum of compilation-inclusive per-run wall time: 23.611 seconds, telemetry only. Four-CPU affinity; sequential runs.
- Agent numeric payload: 60 bytes on SwitchingTwoState, 84 bytes on RiverSwim. Total runner dynamic payload: 88 and 116 bytes respectively, unchanged from initial to final state. This excludes static environment tables, executable memory and allocator overhead. Fixed controls include unused optimizer-state ballast; no memory Pareto claim is made.
- The result field `retained_trajectory_numeric_bytes=80000` describes the reward and TD-error arrays held in memory during each run. Raw arrays are not persisted: the files retain means, RMS error, final weights and reward hashes. Those hashes establish consistency, not authenticated execution proof.
- Verified the complete 40-member environment/arm/seed Cartesian product, absence of duplicate or missing runs, empty failures list, exact step/query counters, unchanged dynamic payload sizes, phase-mean reconciliation, equality to the posted plan, and every recorded Python source digest against the measured checkout.
- The independent recurrence and output-change tests, together with the existing supervised panel, passed (19 tests). Repository Ruff and strict mypy passed before execution.

Runtime: `{"backend": "cpu", "jax": "0.11.1", "jax_enable_x64": false, "machine": "x86_64", "numpy": "2.5.2", "python": "3.12.3"}`.

## Scope

Permanently nonpromoting development evidence. No reference-dev or frozen scorecard change. No tuned-baseline, deep-RL, scientific, robotics, or whole-life benefit claim. The full issue #1561 scope remains open. Do not rerun this exact configuration expecting a win; a new hypothesis and separately recorded plan are required before another comparison.

Result SHA-256: `0d9d0835808200c6a92bda1ebc4f7a47b8fa0377523cbaf2e6d56b678cb7f0e5`.
