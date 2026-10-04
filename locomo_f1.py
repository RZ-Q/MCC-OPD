"""LoCoMo validation rules, extracted without changing their numerical definition.

Adapted from snap-research/locomo/task_eval/evaluation.py (CC BY-NC 4.0).
See THIRD_PARTY_NOTICES.md for attribution and the upstream license.
"""

from collections import Counter
import string

import numpy as np
import regex
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    """规范化整串EM；保留词序、重复和and，不改变官方F1的规范化。"""
    def normalize_em(text):
        text = ''.join(ch for ch in text.lower() if ch not in string.punctuation)
        return ' '.join(regex.sub(r'\b(a|an|the)\b', ' ', text).split())

    return normalize_em(prediction) == normalize_em(ground_truth)

def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1

def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])
