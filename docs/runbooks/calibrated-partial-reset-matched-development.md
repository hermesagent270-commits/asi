# Consolidated CPR matched development contract

Issue [#1563](https://github.com/SlopDotCash/asi/issues/1563) has one authoritative
learner family in `benchmarks/ipmnist_screening.py`. The campaign in
`evaluation/calibrated_partial_reset_campaign.py` consumes those existing
`cpr_off`, `cpr_ipmnist`, `cpr_utility_free`, `cpr_l2_init`, and
`cpr_hard_reset` specs and their strict v2 development receipts. It reuses the
closed #2125 campaign work without restoring its parallel Adam learner.

This is a proposed, permanently nonpromoting development contract. Both
execution literals are hard `False`. No campaign run, retained learning result,
paper reproduction, reference-dev promotion, or scientific evidence is supplied
by this contract or its tests. A separately reviewed authorization transition
and independent audit remain required before execution. The transition must
also update the literal plan digest; changing the gates alone fails before
reservation or dataset discovery.

## Question and protocol

The uncertainty is whether calibrated utility changes improve this supervised
port relative to its exact mechanism-off control. The hypothesis is that
periodic utility-scaled pulls preserve useful parameters while restoring
plasticity in less useful parameters. The five current-source arms are rerun
with matching initialization, schedule, observations, updates, and charged
numeric state. Hard reset, continuous L2-Init, and utility-free pulls distinguish
utility selection from uniform regularization and reset effects.

The fixed development screen uses five proposed roots `1563066001`–`1563066005`,
60 tasks of 5,000 examples, the existing 784–300–150–10 MLP, and canonical OpenML
`mnist_784` version 1 rows 0:60000 with the frozen float32/int32 materialization
hashes in the plan. The primary statistic is mean online-accuracy difference
between `cpr_ipmnist` and `cpr_off`: advance for nonpromoting followup only when
the paired mean exceeds `+0.005` and all five seed differences are positive;
reject when the mean is at or below zero; otherwise retain an inconclusive
outcome. Ablations are descriptive and cannot rescue that rule. Every per-task
metric, paired difference, sample standard deviation, standard error, and
completed negative outcome is retained in the report.

The port pins [arXiv:2607.24996v1](https://arxiv.org/abs/2607.24996) and
[official commit 6fc2af3](https://github.com/LucMc/continual-learning/blob/6fc2af34783159f5dda50c6915dda32c2d443604/continual_learning/optim/cpr.py).
The fetched `continual_learning/optim/cpr.py` has Git blob SHA-1
`b6bd13bbd0a519b122670c19729df8021d815d0d` (also confirmed by the GitHub
contents API at that commit). Its [lines 277–280](https://github.com/LucMc/continual-learning/blob/6fc2af34783159f5dda50c6915dda32c2d443604/continual_learning/optim/cpr.py#L277-L280)
test a positive pre-update clock; initialization at line 164 and increments
at lines 196–198 and 263 imply the first reset at update F+1, then every F
updates. [Lines 256–263](https://github.com/LucMc/continual-learning/blob/6fc2af34783159f5dda50c6915dda32c2d443604/continual_learning/optim/cpr.py#L256-L263)
recenter utility to ones after a reset. Periodic port modes inherit these two
semantics; the plan records them under `upstream_alignment`.

The deliberate adaptations are normalized SGD, per-parameter absolute-gradient
EMA, retained initialization, and pulls of all parameters including biases and
output weights. These differ from the official per-neuron, fresh-reset,
incoming/outgoing weight protocol and its RL workloads. L2-Init acts every
update; mechanism-off never pulls. No task IDs, task-boundary signals,
replay, pretraining, or resets at permutation boundaries are supplied.

The production roster is separate from test-only roots 301–305. Searches of
the visible local trees, all fetched Git histories, and indexed GitHub issue
and PR text found no prior use of the proposed production roots before this
contract. That audit has a visibility limit and must be checked independently
before authorization. These now-exposed roots are forever ineligible for
promotion; old #2125 roots are retired as well. No claim that these are untouched
scientific seeds is made.

## Identity, accounting, and transaction

The report binds every package Python source plus `pyproject.toml` and
`uv.lock`, clean Git source provenance for production, runtime dependencies,
Python/platform/all JAX configuration values and devices, data materialization, registered
hyperparameters, initial parameters, initial learner state and RNG cursor, and
the schedule. Source, runtime, and dataset drift fail closed during both
execution and replay. Hashes bind consistency and do not authenticate execution.

The full transaction charges 25 initial runner calls and 25 strict replay
calls: 15,000,000 observations/data steps/updates and 30,000,000 model queries,
with zero environment steps. Persistent numeric payload includes live
parameters, retained initialization, utility EMA, normalizer, and step counter
for every arm, including unused control buffers. The static 256 MiB combined
envelope additionally charges one contiguous caller-owned host dataset and one
materialized schedule. Logical leaf payload is charged even when initialization
aliases storage. Compiler/backend copies, gradients, transient buffers, Python
objects, and allocator overhead are excluded; this is not a physical RAM bound.
Timing is compilation-inclusive telemetry and does not select an outcome.

The registered new namespace is
`outputs/calibrated_partial_reset_matched/development.v2/`. An exclusive,
fsynced reservation precedes data/source consumers. The complete roster is
marked dispatched/consumed and fsynced before the first learner call. Full
structural validation precedes all replay, which compares every non-timing
receipt field against a fresh dataset-bound runner call. Publication stages
bounded duplicate-key-free JSON in a Linux `O_TMPFILE`, fsyncs it, makes it
read-only, links without replacement, and strictly rereads its owned inode.
Publication is therefore Linux-only: the reservation fails closed with `CPR
publication requires Linux descriptor support` on any host lacking
`O_DIRECTORY`, `O_NOFOLLOW`, or `O_TMPFILE`, and refuses before it opens or
creates any directory entry. The durability tests skip on exactly that
predicate, so they skip where the module would refuse and run where it succeeds.
The held parent must still be the visible registered parent after reservation,
before link, after directory fsync, and during final validation/completion.
Reservation and status writes loop until every byte is written or fail closed.

Pre-dispatch failures release the owned reservation. Post-dispatch failure
retains a consumed-without-result marker when the filesystem permits; completed
reports retain a completion marker, so deleting the report does not permit a
retry. A remaining reservation or marker blocks reuse. Do not remove it to
resume. Ordinary exceptions are handled; process death, failed filesystem
writes, external directory replacement, and durable recovery do not have a
result-retention guarantee. A marker is a disposition, not a measured result.

## Inspect and validate

The v1-to-v2 receipt change fails closed over the old timing contract. At the
reviewed implementation tree, a read-only scan of all 1,318 retained JSON files
under `outputs/` found no CPR-family receipts or v1 schema. A broader scan of
2,098 JSON/source/document/log files also found no CPR arm references. The
13 retained tar/gzip/wheel archives were inspected without extraction; none of
their 1,125 JSON/source/document/log members contained CPR arm or v1 references.
The operator checkout's additional outputs likewise contained no such records.
The only in-tree v1 schema literal is the hostile rejection test in
`tests/test_partial_resets_ipmnist.py`; there is no retained v1 reader to migrate.
The validator reconstructs `partial_reset_development_record(result)` using
the shared `PARTIAL_RESET_RECORD_SCHEMA` and compares the complete payload;
it has no hardcoded v1 acceptance branch. External records remain historical
and must not be rewritten or interpreted as v2 trajectories.

Execution and replay require JAX's default PRNG implementation to be
`threefry2x32` and its random seed offset to be zero; another runtime fails
closed. Each arm's drift check rehashes the complete dataset and enumerates
installed distributions. This overhead is charged as timing telemetry.

The read-only catalog performs no dataset discovery or campaign execution:

```bash
.venv/bin/python -m alberta_framework.evaluation.calibrated_partial_reset_campaign --catalog
```

The normal CLI fails before reservation and default dataset discovery while
either gate is closed. After separate reviewed authorization, it owns the
initial runs, strict replays, and immutable publication:

```bash
.venv/bin/python -m alberta_framework.evaluation.calibrated_partial_reset_campaign --data-home <canonical-openml-cache>
```

`validate_report(..., reexecute=False)` checks structure, identities, accounting,
arithmetic, policy, and digest. Production `reexecute=True` is hard-gated before
dataset, identity, schedule, or learner work. Private test-only execution is
limited to tiny configurations (at most 512 updates, 16 inputs, and 10,000
parameters), uses separate roots, and does not write the registered namespace.
Tests substitute runner callbacks to verify transactions without running a
campaign in pytest. Passing those checks is implementation qualification only.

A positive development outcome would still require recurrence/retention and
downstream reference-agent control checks before any reference channel change.
An authorized execution, retained outcome, independent audit, and integration
remain the open deliverables for #1563.
