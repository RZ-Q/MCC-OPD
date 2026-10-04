# MCC-OPD

Code and data for **Learning When Evidence-Sensitive Guidance Helps: Calibrating
Counterfactual On-Policy Distillation for Memory-Grounded Question Answering**.

MCC-OPD uses a mirror KL budget to calibrate the contrast between full-context
and evidence-ablated teacher predictions. Training uses no answer-level reward.
This package includes the MCC-OPD implementation in verl v0.9.0, one default
training configuration, and four-benchmark scoring and Judge code. The separate
data archive contains the paper's prepared inputs and before/after corrections.

## Setup

Clone this repository and download the [prepared data archive](https://github.com/RZ-Q/MCC-OPD/releases/download/data-v1/MCC-OPD-data.zip).
Extract the archive beside the repository so that its `MCC-OPD/data/` directory
merges into the checkout:

```bash
git clone https://github.com/RZ-Q/MCC-OPD.git
curl -L --fail -o MCC-OPD-data.zip \
  https://github.com/RZ-Q/MCC-OPD/releases/download/data-v1/MCC-OPD-data.zip
unzip MCC-OPD-data.zip
cd MCC-OPD
```

Use Python 3.12 and a CUDA-enabled environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

The paper used one node with 8 NVIDIA A100 GPUs, FSDP with CPU offload, and SDPA
attention. The student is Qwen3-1.7B and the frozen teacher is Qwen3-8B.

## Train

Run from the package root:

```bash
RAY_prestart_worker_first_driver=0 RAY_enable_worker_prestart=0 \
python -m verl.trainer.main_ppo \
  --config-path "$PWD/configs" --config-name mcc_opd
```

The default [configuration](configs/mcc_opd.yaml) uses LoCoMo train140/val77,
N=4, alpha=1.5, top-K=64, seed 45, and three epochs (27 updates). Validation and
checkpoint saving occur every nine updates. The paper reports checkpoint 27.
Checkpoints are written to `outputs/locomo140_n4/checkpoints/`.

To use downloaded model weights, append:

```bash
actor_rollout_ref.model.path=/path/to/Qwen3-1.7B \
distillation.teacher_models.teacher_model.model_path=/path/to/Qwen3-8B
```

## Data

The data archive contains LoCoMo training/validation Parquet files and test
inputs for LoCoMo (1192), MuSiQue (2172), and LongMemEval (499), including the
data before correction and question-level comparisons. REALTALK (728) is
prepared locally from the official download using the included script and
three reference corrections. See [data/README.md](data/README.md) for the command,
formats, revisions, and sources. Dataset files are distributed separately from Git.

## Evaluate

Save model outputs as JSONL with `question_id` and `raw_response`, then run:

```bash
python -m pip install -r requirements-eval.txt
python evaluation/evaluate.py --benchmark locomo \
  --references data/locomo/test.jsonl.gz --predictions answers/locomo.jsonl \
  --output-dir scores/locomo
```

This computes F1, EM, and BLEU-1. Add `--judge` and set `JUDGE_API_KEY` to compute
accuracy with the benchmark rubric, shared question guidance, and two votes
(a third on disagreement). See [evaluation/README.md](evaluation/README.md)
for the four benchmarks and output format.

## Core checks

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q -p no:cacheprovider tests/test_mcc_opd.py
```

## License

Original MCC-OPD code uses [Apache-2.0](LICENSE). Third-party code and datasets
retain their respective licenses, including LoCoMo's CC BY-NC 4.0 terms.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component attribution.
