# Evaluation

Score saved model answers with the paper's four benchmark protocols. The same
entry point computes F1, EM and BLEU-1, and optionally runs the correctness Judge.

From the package root:

```bash
python -m pip install -r requirements-eval.txt
python evaluation/evaluate.py \
  --benchmark locomo \
  --references data/locomo/test.jsonl.gz \
  --predictions predictions/locomo.jsonl \
  --output-dir results/locomo
```

Change `locomo` to `realtalk`, `musique`, or `longmemeval` for the other tracks.
The reference files are in the separate data archive. Each prediction JSONL row
must contain `question_id` and `raw_response`, with exactly one row per reference
ID. `raw_response` is the full generated text, including any `<reasoning>` and
`<answer>` sections. Reference records contain the prepared `messages` and
generation settings for use with an inference runner.

Outputs are `scored.jsonl` with per-question scores on a 0–1 scale and
`summary.json` with aggregate percentages, including per-category results.
The summary also records the reference and prediction file hashes.

## Correctness Judge

Set `JUDGE_API_KEY` in your environment and append `--judge` to the same command.
The default endpoint is `https://api.deepseek.com/chat/completions`, with model
alias `deepseek-flash` (the paper's DeepSeek-V4.1-Flash configuration), temperature
0, disabled thinking, and a 2048-token output limit. An endpoint or model can be
selected with `--judge-url` and `--judge-model`; the chosen values are recorded
in the output.

Each nonempty answer receives two independent calls. Disagreement triggers a
third call and majority voting. Identical complete Judge inputs share the same
votes. Empty extracted answers receive zero. Even an exact reference match
receives the two calls. The Judge sees the final answer after case folding,
whitespace folding and removal of trailing sentence punctuation; lexical
metrics use the extracted answer before this canonicalization.

`votes/` saves the prompt, model and individual raw responses. Rerun the same
command to resume completed votes after an interrupted request. Failed or
unparseable votes remain missing and make aggregate Judge accuracy unavailable
until completed. `--workers` controls request concurrency (default 16).

The LoCoMo rubric is REMem/Mem0-style, MuSiQue and REALTALK use the project's
reference-semantic rubrics, and LongMemEval uses its official task-specific
rubrics, including abstention. [rubrics.py](rubrics.py) contains the full text.
The fixed question-level guidance is in [guidance/locomo.json](guidance/locomo.json)
and [guidance/longmemeval.json](guidance/longmemeval.json), and applies to every
model. LongMemEval additionally reports the equally weighted mean across six
task accuracies and accuracy on its 30-question abstention subset.

## Answer extraction and lexical metrics

The evaluator uses the shared [answer extractor](../reasoning_answer.py).
LoCoMo and LongMemEval normalize complete answer/reasoning boundary tags before
extraction; REALTALK and MuSiQue use the original tag spelling. Strict format
validity is recorded separately from answer correctness. The paper's final
results use the prepared answers after one additional generation attempt for
empty extractions; this scorer operates on the supplied answers and performs
no model generation.

- **LoCoMo and REALTALK F1:** lowercase, remove punctuation and `a/an/the/and`,
  split on whitespace, and apply Porter stemming. Category 1 averages the best
  predicted comma-separated item match for each reference item; categories
  2–4 use whole-answer token F1. Category 3 truncates the reference at its first
  semicolon. REALTALK takes the best score across accepted alternatives.
- **MuSiQue F1:** token overlap after lowercasing and removing punctuation,
  articles and extra whitespace, with the best score over reference aliases.
- **LongMemEval lexical F1:** the paper's fixed 2Wiki/HotpotQA token-overlap
  rules, using a single reference. Normalization removes punctuation,
  `a/an/the` and extra whitespace; a mismatched `yes`, `no` or `noanswer` has
  zero F1. Its task correctness is reported separately by the official Judge.
- **EM:** normalized whole-string equality, preserving word order and repeated
  words; no stemming. The normalization keeps `and` on all four benchmarks.
- **BLEU-1:** case-sensitive whitespace tokens, NLTK sentence BLEU with
  weights `(1, 0, 0, 0)`, method-1 smoothing and the brevity penalty. Benchmark
  scores average per-question scores.

## Sources

- LoCoMo F1: [snap-research/locomo](https://github.com/snap-research/locomo),
  reused through [locomo_f1.py](../locomo_f1.py); CC BY-NC 4.0, as recorded in
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- LoCoMo Judge text: [REMem](https://github.com/intuit-ai-research/REMem),
  revision `d87840be922c9815a3c97ec24d9c25e4af5c295a`,
  `src/remem/evaluation/qa_mem0_llm_judge.py`; [Apache-2.0](licenses/Apache-2.0.txt).
- MuSiQue answer metrics: [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique),
  revision `922ac98f19a201998dbdae6d7f2887a5258dbdeb`;
  [CC BY 4.0](licenses/MuSiQue-CC-BY-4.0.txt).
- LongMemEval Judge prompt: [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval),
  revision `9e0b455f4ef0e2ab8f2e582289761153549043fc`;
  [MIT](licenses/LongMemEval-MIT.txt).
- LongMemEval lexical metrics: the `normalize_answer`, `f1_score` and
  `exact_match_score` functions from
  [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)'s
  `2wikimultihop_evaluate_v1.1.py` (adapted upstream from HotpotQA);
  [Apache-2.0](licenses/Apache-2.0.txt).
