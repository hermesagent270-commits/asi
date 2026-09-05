# Intentional TD development comparison

Issue [#1561](https://github.com/SlopDotCash/asi/issues/1561) requests both
supervised and TD/control Intentional Updates. The existing IPMNIST family
implements the supervised extension. `core/intentional_td.py` adds the value
optimizer, and `benchmarks/intentional_td_development.py` consumes it in an
uninterrupted linear SARSA agent on the existing SwitchingTwoState and
RiverSwim environments.

The [completed development comparison](../../outputs/intentional_td_development_v1/REPORT.md)
failed its preregistered benefit rule: the full arm underperformed the
fixed-step trace control in mean reward on both environments. All 40 runs and
all four development seeds are retained. This does not close the issue's
complete empirical scope or modify the frozen 144-shard scorecard or
`reference-dev`.

## Reference and deviations

The reference is [Intentional Updates, v1](https://arxiv.org/abs/2604.19033v1)
and `sharifnassab/Intentional_RL` at
`e86e26fd8613ac212e9a52c3fed8a01d0a31f685`, specifically
`optimizer.py:IntentionalOptimizerValue`. The implementation uses the
reference code's bias-corrected discounted sigma **mean**. Paper Eq. 12 instead
describes an unnormalized discounted sum; those are different update scales.
The independent NumPy recurrence in `tests/test_intentional_td.py` checks the
code-defined update and terminal trace reset.

The consumer uses linear action values and on-policy SARSA targets. It does
not reproduce the paper's deep networks, Q-learning, policy-gradient methods,
or external-environment results. There is no replay, pretraining, task ID,
phase-switch reset, or privileged oracle input.

## Proposed comparison

The CLI's `--plan` output is authoritative for all parameters:

```bash
.venv/bin/python -m alberta_framework.benchmarks.intentional_td_development --plan
```

- Development seeds: 15610, 15611, 15612, 15613. No tuning or held-out claim.
- Five arms: full intentional, no trace, no RMS, fixed-step trace, fixed-step
  without trace. Each starts from zero weights on the same observation basis.
- Two environments: SwitchingTwoState with 500-transition phases and
  three-state RiverSwim. Four seeds × five arms × two environments gives
  40 runs, each with 10,000 transitions and updates.
- The random keys and epsilon-greedy policy rules match; learned actions and
  visited states can differ. Each life continues across all payoff switches.
- The metric is reward per transition, including all early learning. Retain
  per-phase means, pre-update TD errors, final weights, counters, source
  hashes, and every failure. Re-measure both fixed controls in the same run.
- Call the result a development benefit only if the full intentional arm
  exceeds both fixed controls by more than 0.02 mean reward in both
  environments and every paired seed delta is positive. Otherwise report
  negative or inconclusive. This criterion never promotes an arm.

No historical result exists for this exact comparison. The intended benefit
is adaptive value-unit updates with temporal credit assignment. The no-trace
and no-RMS arms distinguish those components. Fixed controls retain unused
optimizer arrays as explicit accounting ballast; they are not minimal-memory
implementations. The output separately reports actual agent array bytes and
runner dynamic array bytes. Static environment tables and executable memory
are excluded. Compilation-inclusive timing is telemetry, not a compute-parity
or latency result. `prediction_queries` counts logical vector or scalar
value-evaluation calls before compiler common-subexpression elimination.

After the plan is publicly recorded on the existing issue, execute to a NEW
path and retain the complete log:

```bash
.venv/bin/python -m alberta_framework.benchmarks.intentional_td_development \
  --output <new-development-result.json>
```

All results remain development-only and permanently nonpromoting. The
supervised comparison, deep-RL reproduction, stronger tuned baselines,
whole-life adapter, and complete issue acceptance remain separate open work.
