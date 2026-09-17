# TeLAPA smoke replay reconciliation

The append-only [review artifact](review_reconciliation.v1.json) binds all
three retained v2 result files by their full-file SHA-256 and by the canonical
SHA-256 of their `records` arrays. All three record-array digests are
`4195b4d6379e6cfb54cc13de341a038e41945a2dae36d2d0b1a05091d189564c`.
Canonicalization uses UTF-8 JSON with sorted keys, compact separators and
`allow_nan=False`, preserving array order. Duplicate keys and non-finite
numbers are rejected before comparison.

There are 24 historical/current comparisons: each of the twelve `(seed, arm)`
pairs against each of the two historical files. Each comparison includes both
record digests and 27 individual field matches, including every resource
receipt field. Equality covers all record fields; it does not imply equal
historical source or runtime identities, independent seed trajectories, or
authenticated execution.

## Why the seed trajectories coincide

The existing runner creates a root key from each seed. Adapter initialization
uses `fold_in(root, 0)` for the environment's initial state and
`fold_in(root, 1)` for its initial 2x2 policy. The audit reconstructs those
initial values with explicit Threefry keys and checks the policy digest against
every retained arm for that seed:

| Seed | Initial state | Initial greedy actions for states 0, 1 |
|---|---:|---|
| 1586000 | 1 | 0, 0 |
| 1586001 | 0 | 1, 1 |
| 1586002 | 1 | 0, 0 |

All three initial policy digests differ. All three final policy digests also
differ within each arm. The policy uses `argmax` without stochastic action
sampling. Different initial weights therefore need not change any greedy
decision. Seeds 1586000 and 1586002 have the same initial state and greedy
choices and produce the same action and observation hashes in **all four**
arms. Seed 1586001 produces different action and observation hashes. The
strict validator independently reconstructs every complete record, including
these trajectory hashes and final policy hashes.

After initialization, SwitchingTwoState transitions are deterministic:
`next_state = action`. The execution keys are passed through the adapter but
the environment step explicitly ignores them. Rewards depend on the current
state, action and scheduled phase. Both default payoff matrices are invariant
under simultaneously complementing state and action, so complementary
trajectories can receive identical rewards. Reward hashes coincide across
seeds within each arm; rewards are action-dependent and differ between arms.
These three roots do not establish three independent environment-randomness
trials or robustness across diverse trajectories. The short smoke supports
only its retained, permanently nonpromoting consistency and negative outcome.

## Validator record and historical rejection reasons

The [audit script](review_audit.v1.py) calls the existing v2 `validate_result`
without replacing identities or weakening gates. Successful validation
reexecutes all twelve 32-step records from their bound configurations. The
machine artifact records the command, validator schema/source hash, audit
source hash, validated source/runtime/dependency identity, JAX PRNG/x64
configuration, observed success status and exact historical rejections.

The original `development_result_pr2257.v2.json` differs in policy-archive and
environment source hashes and JAX/NumPy versions (0.11.1/2.5.2 versus
0.11.0/2.5.1); its Python version is already 3.12.3. The later
`development_result_pr2257_review.v2.json` instead differs only in Python
version (3.12.14 versus 3.12.3). Both correctly fail the current identity gate
before record reexecution. Their records match the current-runtime replay,
but neither historical identity is rewritten or adopted as current evidence.

From the repository root on the bound Linux x86_64 project runtime:

```bash
PYTHONPATH=. /root/asi/.venv/bin/python outputs/telapa_qualification/review_audit.v1.py --check
```

This regenerates the report in memory, reexecutes the strict validator and
requires exact equality with the retained review artifact. It writes no result
file. Omitting `--check` creates the review artifact only at an unoccupied
path. The import guard prevents accidentally validating an installed checkout.
The check fails closed if bound source, runtime, dependencies, inputs, audit
version or report values drift. The hashes are consistency bindings, not
authenticated execution attestation.

The original replay was produced at main
`f3d32c451ed1c1715e477ad56b782fc7ea89b206`. That remains its historical snapshot;
the checker evaluates the live lane identity at the checked head rather than
assuming any later main or merge head still matches. A merge changing bound
sources must pass the check again on the bound runtime or retain this record
as historical. Existing results remain byte-identical. No prospective seeds,
paper implementation, scientific evaluation or performance promotion is added.
