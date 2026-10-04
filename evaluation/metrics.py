"""四轨论文指标；输入为已经抽取的最终答案，不再次抽取。"""

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from locomo_f1 import exact_match_score, f1, f1_score
from evaluation import longmemeval_metrics, musique_metrics


def score_answer(benchmark: str, reference: dict, answer: str) -> dict:
    if benchmark in {"locomo", "realtalk"}:
        category = int(reference["category"])
        if category not in {1, 2, 3, 4}:
            raise ValueError(f"unsupported category: {category}")
        golds = ([*reference.get("accepted_answers", [reference["answer"]]),
                  *reference.get("answer_aliases", [])] if benchmark == "realtalk"
                 else [reference["answer"]])
        golds = [str(g).split(";", 1)[0].strip() if category == 3 else str(g) for g in golds]
        score_f1 = max((f1 if category == 1 else f1_score)(answer, g) for g in golds)
        score_em = max(exact_match_score(answer, g) for g in golds)
    elif benchmark == "musique":
        golds = list(dict.fromkeys([str(reference["answer"]),
                                   *reference.get("answer_aliases", [])]))
        score_f1 = max(musique_metrics.compute_f1(answer, g) for g in golds)
        score_em = max(musique_metrics.compute_exact(answer, g) for g in golds)
        # BLEU 沿用汇总脚本接受的 reference 集合。
        golds = list(dict.fromkeys([str(reference["answer"]),
                                   *reference.get("accepted_answers", []),
                                   *reference.get("answer_aliases", [])]))
    elif benchmark == "longmemeval":
        golds = [str(reference["answer"])]
        score_f1 = longmemeval_metrics.f1_score(answer, golds[0])[0]
        score_em = longmemeval_metrics.exact_match_score(answer, golds[0])
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    bleu = sentence_bleu(
        [g.split() for g in golds], answer.split(), weights=(1, 0, 0, 0),
        smoothing_function=SmoothingFunction().method1,
    )
    return {"f1": float(score_f1), "em": float(score_em), "bleu1": float(bleu)}
