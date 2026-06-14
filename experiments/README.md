# experiments/

Runnable experiments and notes. The training/eval pipeline lives in the package;
run it as a module from the repo root.

- **Train an exact model** (saves to `../models/`):
  ```bash
  python -m vcirc.train --n 10 --seeds 6
  ```
- **Scaling check** (other lengths):
  ```bash
  python -m vcirc.train --n 12 --seeds 3
  ```

Foothold-A1 results (n=10 and n=12, all seeds exact 100%) are summarised in
`../docs/PROGRESS.md`.
