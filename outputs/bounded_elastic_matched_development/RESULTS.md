# Bounded elastic matched development result

This permanently nonpromoting issue #1562 campaign ran at exact source commit
`8f6cea1705f774eab61272d3afd636265010c588`. The preregistered plan digest is
`4c7a991acf81f3846c00b73e55d8e80bf6e8b23486996a8523e6a6000e70e02e`, and the
immutable result's canonical digest is
`17681f37d707afa90276672ff59bb7420c896de4b5a43f43756176bd176fc613`.

The transaction completed all 20 initial rows and all 20 strict reexecution rows: four arms,
five fresh development seeds, eight tasks per row, and 5,000 examples per task. Across both
passes it charged 1,600,000 observations, optimizer updates, and data steps plus 3,200,000 model
queries. The runner validated the same source, runtime, dataset, schedule, and initialization
identities before publishing one create-only artifact.

## Decision

The campaign is **rejected** under its frozen sign rule. Fixed-capacity CBP achieved mean online
accuracy `0.751540` (sample SD `0.001894`). Bounded growth achieved `0.728955` (sample SD
`0.005060`), for a paired delta of `-0.022585` (sample SD `0.003910`, SE `0.001749`). Bounded
elastic achieved `0.728820` (sample SD `0.004623`), for a paired delta of `-0.022720` (sample SD
`0.003515`, SE `0.001572`). All five paired deltas were negative for both candidates. The
descriptive structure-off arm measured `0.728870` (sample SD `0.004805`).

The bounded arms stayed inside the shared `1,132,248`-byte peak persistent budget. Bounded growth
ended with 158 active first-layer units and `597,560` active parameter bytes; bounded elastic
ended with 150 units and `567,640` bytes; fixed CBP used all 300 units and `1,128,640` active
parameter bytes. The compact candidates therefore met the registered resource constraints but
did not match the stronger control's online accuracy at this horizon.

Initial-execution timing telemetry totaled `46.25` seconds for structure-off, `47.55` for bounded
growth, `47.92` for bounded elastic, and `139.37` for fixed CBP across five seeds. Timing was not
part of the decision rule, and strict-reexecution timing is not stored in the result rows.

## Audit and scope

[`audit.py`](audit.py) independently reloads strict JSON, verifies the plan and result digests,
recovers all seven registered source files from the measured Git commit, checks the exact 20-row
Cartesian roster and matched execution identities, validates resource receipts, and recomputes
every arm aggregate, paired delta, sign decision, and the campaign outcome. Its retained output
is [`audit.v1.json`](audit.v1.json). The audit script SHA-256 is
`fbc420bdb06951b8518770cca745aefe1964f98efd143bd9e4836bd28e7c364c`.

Preflight found and fixed one blocker before any campaign dispatch: the canonical OpenML loader
could preserve Fortran order while the frozen transaction requires a caller-owned C-contiguous
dataset. Commit `b465b03b` added the failing regression and materialized the same registered bytes
in C order. No seed was consumed by that preflight.

This result rejects only ASI's registered fixed-shape IPMNIST adaptation at this configuration and
horizon. It is not a reproduction or rejection of the paper's dynamically deep method, a
scientific result, a state-of-the-art claim, or evidence for promotion. Consistency hashes are not
authenticated execution attestation.
