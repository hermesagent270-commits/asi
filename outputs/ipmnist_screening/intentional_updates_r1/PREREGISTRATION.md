# Preregistration — supervised Intentional Updates development screen

**Written 2026-09-16 before any campaign shard was executed.** This is a
development screening diagnostic and is permanently nonpromoting.

## Question and hypothesis

Issue #1561 asks for batch-size-one supervised and TD/control Intentional
Updates under matched update budgets. The TD/control slice is retained in PR
#2889. This campaign closes the supervised measurement gap around the already
registered IPMNIST implementation.

The primary hypothesis is that the full supervised Intentional Updates arm
improves mean online accuracy over its exact mechanism-off reduction by more
than 0.005 at the 60-task horizon, with every paired seed improving. The causal
mechanism is the intended-output-change step size, using RMSProp diagonal
normalization and adaptive clipping. The ablations test whether diagonal
normalization, clipping, or feature updates account for any observed effect.

## Frozen protocol

- Paper: `arXiv:2604.19033v1`.
- Official code: `sharifnassab/Intentional_RL@e86e26fd8613ac212e9a52c3fed8a01d0a31f685`.
- Protocol extension: supervised IPMNIST with batch size one, `lambda=0`, and
  correct-class log probability as the output target. This is not a
  reproduction of the publication's RL experiments.
- Horizon: 60 permutation tasks × 5,000 examples, exact `step` mode.
- Development seeds: `1561000, 1561001, 1561002, 1561003, 1561004`.
- Arms: `intentional_updates_ipmnist`, `intentional_updates_no_diag`,
  `intentional_updates_no_clip`, `intentional_updates_head_only`, and
  `intentional_updates_off`.
- Primary control: `intentional_updates_off`, the bit-exact fixed-step
  normalized-SGD reduction.
- Matched axes: seed, example schedule, observations, updates, backward passes,
  model queries, and absence of task/boundary information.
- Primary metric: mean online accuracy over all 60 tasks.
- Secondary descriptive metrics: mean loss, mean plasticity, and the 15-task
  late-window accuracy slope.
- Resources: every shard records observations, updates, backward passes, model
  queries, persistent numeric bytes, and compilation-inclusive wall time.
  Timing is telemetry only.

All 25 shards and every failure are retained. No tuning or replacement run is
allowed after seeing these seeds.

## Frozen decision rule

For the full arm versus the mechanism-off control:

- **development win:** paired mean online-accuracy delta is greater than
  `+0.005` and all five paired seed deltas are positive;
- **development rejection:** paired mean delta is at most `0.0`;
- **inconclusive:** every other outcome, including a positive mean that misses
  the margin or has any nonpositive seed.

A win remains nonpromoting and does not select `reference-dev`. Ablations are
reported descriptively and cannot rescue a failed primary gate.

## Failure conditions

The hypothesis fails if the primary gate does not pass. The campaign is invalid
if any arm/seed is missing, source/dataset/runtime identities differ, a record
fails the strict Intentional Updates validator, or matched resource counters
differ. Invalid execution is retained and is not interpreted as performance.

