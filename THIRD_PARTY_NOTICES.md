# Third-party notices

Original MCC-OPD code is licensed under Apache-2.0 (see `LICENSE`). The following
third-party components and datasets retain their own licenses.

- `verl/`: verl v0.9.0, revision
  `483b8a009ba3a97563edee3a19887e4862b8094a`, with MCC-OPD training changes.
  Original repository: https://github.com/verl-project/verl.
  Retain `verl/LICENSE` (Apache-2.0), `verl/Notice.txt`, and source headers.
- `mcc_opd.py` and `verl/verl/trainer/distillation/mcc_core.py`: MCC-OPD tensor
  implementation, retaining the Apache-2.0 notice of the adapted scoring code.
- `locomo_f1.py`: adapted LoCoMo category-aware F1 scoring from
  https://github.com/snap-research/locomo, CC BY-NC 4.0.
  See `licenses/locomo.txt`. Used for validation and held-out lexical evaluation.
  The short-answer instruction in `data/prepare_realtalk.py` also follows the
  LoCoMo prompt, with MCC-OPD's reasoning wrapper.
- `evaluation/`: benchmark scoring functions, Judge prompts, and the paper's
  question-specific guidance. See `evaluation/README.md` and the accompanying
  license texts for their upstream sources.
- Prepared datasets retain their upstream attribution. See `data/README.md`
  and the license texts accompanying the separate data archive.
