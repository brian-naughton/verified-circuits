# Milestone B kickoff plan — Lean 4 proof `Circuit == Spec`

**For this session (and any that resumes it).** Project #3 ("verified-circuits");
A1 + A2 are done and merged to `main`. Orient from `MEMORY.md` + `README.md` +
`docs/design.md` + `docs/PROGRESS.md` + `docs/A2-PLAN.md`. A2 shipped the harder,
empirical half — `Circuit == Model` argmax on every length-10 input, by rigorous
interval arithmetic, re-checkable torch-free. **Milestone B is the Lean half:** a
kernel-checked proof that the extracted circuit equals the Dyck-1 spec. Landing it
completes the chain `Spec == Circuit == Model` and is the precondition (with the
n=16 scale-up) for going public.

Recommended execution: **this session as mission control**, a subagent for the
heavy/bounded toolchain install, Codex as the faithfulness adversary, and TDD-by-
build for the proof itself (`lake build` green + a clean `#print axioms` is the
test). Keep `main` clean — all work on branch `b-lean-circuit-spec`.

## Goal

A Lean 4 theorem, proved **by structural induction over the input (not by
enumerating 2ⁿ cases)**, hence holding for **every** length n (n=16 included for
free):

```lean
theorem circuit_eq_spec (s : List Tok) : Circuit.valid s = Spec.isValid s
```

`lake build` green = the artifact. A third party then verifies `Circuit == Spec`
by trusting only the Lean kernel. The companion deliverable is honesty
infrastructure: a faithfulness map from each Lean definition to the exact Python
lines it mirrors, plus a `#eval` cross-check that the Lean `Spec`/`Circuit` agree
with `vcirc/dyck.py` over small full domains (guards against "proved two *wrong*
things equal").

## The mathematics (so the proof is scoped before any Lean is written)

Both sides are the conjunction `finalDepth == 0  ∧  <order condition>`, over the
same running-depth sequence. Two facts carry the whole theorem:

1. **Same final depth.** Circuit's `final_depth` (the running accumulator `dₙ`)
   equals Spec's `final_depth` (`Σ step t`). Trivial: the accumulator *is* the
   running sum. Proven by induction generalising the start depth.

2. **Order conditions coincide.** `violation_count s == 0  ⟺  minPrefixDepth s ≥ 0`.
   Both are equivalent to **"every running prefix depth is ≥ 0"**:
   - `violation_count == 0  ⟺  ∀ i, ¬(dᵢ < 0)  ⟺  ∀ i, dᵢ ≥ 0` (a sum of 0/1
     flags is zero iff every flag is zero).
   - `minPrefixDepth = min(0, d₁,…,dₙ) ≥ 0  ⟺  ∀ i, dᵢ ≥ 0` (the leading `0 ≥ 0`
     is free; the min is ≥ 0 iff every element is).

   Cleanest Lean shape: a single accumulator-generalised predicate
   `allNonneg (d0 : ℤ) : List Tok → Prop` (`[] ↦ True`; `t::ts ↦ d0+step t ≥ 0 ∧
   allNonneg (d0+step t) ts`), prove **both** order conditions equal `allNonneg 0 s`
   by induction on `s` with `d0` generalised, then chain.

The arithmetic at each inductive step is linear over ℤ — `omega` territory. There
is no enumeration anywhere; the proof is length-generic by construction.

## B0 — toolchain (subagent; heavy, bounded, runnable in background)

Install `elan` (→ `lean`/`lake`) on macOS and stand up a Lake project under
`proofs/`. **Dependency decision (settled by mission control 2026-06-14):
minimal-first.** The proof is elementary integer/list induction and `omega` ships
in Lean core, so the dependency ladder is:

1. **Lean core + `omega`** (the target) → 2. **prove the missing 3-line `List`/
order helper inline by induction** (stay Mathlib-free) → 3. **Batteries** (small
dep, fine if it saves real List plumbing) → 4. **Mathlib** (genuine last resort,
unlikely ever reached).

Rationale: the dev loop is rebuild-constantly, so seconds-long builds vs a 30-min
Mathlib compile is a large iteration-speed win on an elementary proof; a busy
reviewer will actually `lake build` a self-contained seconds-to-compile artifact
(low friction = more people verify it = more attention — same instinct as A2's
torch-free stdlib checker); and "kernel + a few hundred lines of our own Lean, no
Mathlib" is an on-brand, auditable, self-contained trust surface.

**Honesty discipline (must be in the writeup):** minimal-deps is **not more
sound** than Mathlib — Lean's kernel checks Mathlib too, so using it would not
weaken the proof. The wins are build-speed, reproducibility, and a smaller
auditable artifact, **not** soundness. State it that way so a formal-methods
reviewer doesn't read "Mathlib-free" as "Mathlib = less trustworthy" (it isn't;
it's just heavier). Pin the `lean-toolchain` to whatever the chosen rung requires.

Proposed layout:

```
proofs/
  lean-toolchain            # pinned Lean version
  lakefile.toml (or .lean)  # package + (maybe) Batteries/Mathlib dep
  lake-manifest.json        # generated
  VerifiedCircuits.lean     # root: imports the modules below
  VerifiedCircuits/
    Spec.lean               # mirrors vcirc/dyck.py
    Circuit.lean            # mirrors vcirc/circuit.py
    Equiv.lean              # lemmas + theorem circuit_eq_spec
```

**Seam 1 (check in):** `elan`/`lake` installed (report versions); an empty
package and a skeleton with the `Spec`/`Circuit` definitions and the theorem
stated as `:= by sorry` both `lake build` cleanly. Decide minimal-deps vs Mathlib
here based on what actually resolves.

## B1 — faithful definitions (mission control; small, high-stakes)

Translate the two Python algorithms into Lean **line-for-line faithfully** — this
is where a sloppy translation would silently prove the wrong thing, so it is owned
here, not delegated:

- `Tok` (inductive `opn | cls`, or `Bool`); `step : Tok → ℤ` (`opn ↦ +1`,
  `cls ↦ -1`).
- `Spec` (mirror `dyck.py`): `finalDepth`, `minPrefixDepth` (running min, init 0),
  `isValid := finalDepth == 0 ∧ minPrefixDepth ≥ 0`.
- `Circuit` (mirror `circuit.py`): the `trace` fold producing per-position depths
  + violation flags, `finalDepth`, `violationCount`, `valid := finalDepth == 0 ∧
  violationCount == 0`.

Each definition carries a doc-comment citing the exact `vcirc/dyck.py` /
`vcirc/circuit.py` lines it mirrors. Add a `#eval`/`#guard` block enumerating
n ≤ ~12 that checks Lean `isValid`/`valid` reproduce the Python truth table
(Catalan counts), so faithfulness is corroborated, not asserted. **Codex
adversary pass here:** is each Lean def a faithful image of its Python source? Is
the theorem statement the meaningful one (not vacuous / not true for the wrong
reason)? (See the `codex-adversarial-agent` memory: pinned model, ≤2.5 KB brief,
≤3 questions, deterministic output path.)

## B2 — the proof (TDD-by-build; individual lemmas may go to subagents)

Implement the decomposition above in `Equiv.lean`:

1. `finalDepth_circuit_eq_spec` — accumulator generalised; induction; `omega`.
2. `violationCount_zero_iff_allNonneg` — induction, `d0` generalised.
3. `minPrefixDepth_nonneg_iff_allNonneg` — induction, `d0` generalised.
4. `circuit_eq_spec` — combine (1)+(2)+(3); both sides are the same conjunction.

Each lemma is independently checkable (its own `lake build`), so a stuck tactic
proof can be handed to a subagent with a crisp contract ("make this `theorem …`
compile without `sorry`/`native_decide`; return the proof term"). Mission control
owns integration and statement correctness.

**Seam 2 (check in):** lemmas (2)+(3)+(1) compile without `sorry`; only
`circuit_eq_spec`'s final assembly may remain.

## B3 — green build + axiom audit + trust story

- `lake build` green, **no `sorry`**.
- `#print axioms circuit_eq_spec` — must show only standard kernel axioms
  (`propext`, `Classical.choice`, `Quot.sound`). **Must NOT show `sorryAx`**, and
  (since we want the inductive, not the enumeration, proof) **must NOT show
  `Lean.ofReduceBool`** (that would mean `native_decide` crept in). Capture the
  output in the docs — it is the real evidence the proof is what we claim.
- Update `proofs/README.md` (reproduce: `cd proofs && lake build`; one-line trust
  statement) and `docs/PROGRESS.md` (a "Milestone B" section in the A2 style:
  claim, method, what's trusted, verdict). Update the README status table row B → ✅.
- **Codex adversary pass (final):** any hidden `sorry`/`axiom`/`native_decide`?
  Is `circuit_eq_spec` non-vacuous and faithful to the Python? Does it genuinely
  hold for all lengths (not silently fixed-n)?
- Commit on `b-lean-circuit-spec`; ff-only merge to `main` (mirroring A2). Public
  remote/PR stays **held** until n=16 is also done (per memory decision).

**Seam 3 (check in):** `lake build` green + clean `#print axioms` pasted →
Milestone B done; `Spec == Circuit == Model` chain complete.

## Fallbacks (in preference order)

1. **Inductive proof, Lean core + `omega`** — the target above.
2. **Inductive proof, missing helper proved inline** — if core lacks a `List`/
   order lemma, prove the 3-line helper by induction and stay Mathlib-free.
3. **Inductive proof + Batteries** — small dep, fine if it saves real List
   plumbing; same theorem strength.
4. **Inductive proof + Mathlib** — genuine last resort (heavier build, same
   theorem strength; Lean's kernel checks it just the same — no soundness loss).
5. **Per-n `decide`** — finite kernel check for specific n; weaker (fixed lengths,
   not length-generic) but still kernel-trusted. Document the downgrade.
6. **Per-n `native_decide`** — last resort; faster but introduces the
   `Lean.ofReduceBool` trust caveat (trusts the compiler, not just the kernel).
   Must be flagged loudly in the README if used.

## Kill / pivot criteria

- The inductive proof devolves into enumeration with no length-generic statement
  achievable → drop to fallback 3/4 and **state the weaker claim honestly**
  (carried from `docs/design.md` kill criteria).
- The Lean `Spec` cannot be written to faithfully match `dyck.py` (truth tables
  diverge on the `#eval` cross-check) → stop; the formalisation, not the proof, is
  wrong.

## After B

Scale to n=16 (the headline 65,536-input domain) for the A2 `Circuit == Model`
certificate; complete the pre-publish checklist in the project memory (v2 ⊇ v1
enclosure-soundness assertion; reviewer FAQ); then go public.
