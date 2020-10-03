from typing import Any, Callable, Dict, List

import numpy as np
from allennlp_eraser.training.metrics.position_scored_document import (
    PositionScoredDocument,
)
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def _auprc(truth: Dict[Any, List[bool]], preds: Dict[Any, List[float]]) -> float:
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    aucs = []
    for k, true in truth.items():
        pred = preds[k]
        true = [int(t) for t in true]
        precision, recall, _ = precision_recall_curve(true, pred)
        aucs.append(auc(recall, precision))
    return np.average(aucs)


def _score_aggregator(
    truth: Dict[Any, List[bool]],
    preds: Dict[Any, List[float]],
    score_function: Callable[[List[float], List[float]], float],
    discard_single_class_answers: bool,
) -> float:
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    scores = []
    for k, true in truth.items():
        pred = preds[k]
        if (all(true) or all(not x for x in true)) and discard_single_class_answers:
            continue
        true = [int(t) for t in true]
        scores.append(score_function(true, pred))
    return np.average(scores)


def score_soft_tokens(paired_scores: List[PositionScoredDocument]) -> Dict[str, float]:
    truth = {(ps.ann_id, ps.docid): ps.truths for ps in paired_scores}
    pred = {(ps.ann_id, ps.docid): ps.scores for ps in paired_scores}
    auprc_score = _auprc(truth, pred)
    ap = _score_aggregator(truth, pred, average_precision_score, True)
    roc_auc = _score_aggregator(truth, pred, roc_auc_score, True)

    return {
        "auprc": auprc_score,
        "average_precision": ap,
        "roc_auc_score": roc_auc,
    }
