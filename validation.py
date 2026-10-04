"""Category-aware LoCoMo validation F1; MCC training does not call a reward model."""

from pathlib import Path
import sys

# verl loads this callback by file path inside Ray workers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomo_f1 import exact_match_score, f1, f1_score
from reasoning_answer import extract_answer


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    extraction = extract_answer(solution_str, normalize_boundaries=True)
    answer = str(extraction["answer"])
    category = int(extra_info["category"])
    if category not in {1, 2, 3, 4}:
        raise ValueError(f"LoCoMo validation excludes category {category}")
    gold = str(ground_truth)
    if category == 3:
        gold = gold.split(";", 1)[0].strip()
    score = float(f1(answer, gold) if category == 1 else f1_score(answer, gold))
    return {
        "score": score,
        "answer_f1": score,
        "answer_em": float(exact_match_score(answer, gold)),
        "answer_extract_valid": float(bool(answer.strip())),
        "format_valid": float(extraction["format_valid"]),
    }
