# Progress

## 2026-06-14 — Foothold-A1: exact model ✅

**Question:** can a tiny, transparent transformer learn Dyck-1 validity to
*exact 100%* over the entire finite domain (with positive decision margins), so
there is something exactly-certifiable to extract a circuit from?

**First attempt (mean-pool, bidirectional, 1 layer):** plateaued at ~99.8–99.9%.
Across an 8-seed sweep the best had 1 error; *every* error was the order /
min-prefix check (`valid-acc` always 100%; some invalid strings classified
valid). The architecture could count depth but not *exactly* test "did a prefix
go negative?".

**Redesign (causal · 2 layers · sum-pool · MLP head):** principled inductive bias —
causal attention computes per-position prefix depth, the second layer detects
per-position violations, sum-pool aggregates final depth + violation count, the
MLP head forms the `|final depth| == 0` test.

**Result:**

| n | domain | seeds at exact 100% | min margin range |
|---|---|---|---|
| 10 | 1,024 | 6 / 6 | +5.7 … +7.9 |
| 12 | 4,096 | 3 / 3 | +4.5 … +7.5 |

Robust (every seed), not fragile; healthy margins (decision-level certification
has ample room). The principled redesign working first try is itself a signal
that the model implements the depth→violation→aggregate decomposition — i.e. an
*extractable* circuit. Exact checkpoints saved under `models/`.

**Verdict:** A1 passed. Next: A2 — extract the symbolic circuit and certify
`Circuit == Model` argmax on every input.

Reproduce: `python -m vcirc.train --n 10 --seeds 6`.
