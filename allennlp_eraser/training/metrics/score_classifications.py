from typing import Dict, List

import numpy as np
from allennlp_eraser.common.util import Annotation
from allennlp_eraser.training.metrics.compute_aopc_scores import compute_aopc_scores
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, classification_report


def score_classifications(
    instances: List[dict],
    annotations: List[Annotation],
    docs: Dict[str, List[str]],
    aopc_thresholds: List[float],
) -> Dict[str, float]:
    def compute_kl(cls_scores_, faith_scores_):
        keys = list(cls_scores_.keys())
        cls_scores_ = [cls_scores_[k] for k in keys]
        faith_scores_ = [faith_scores_[k] for k in keys]
        return entropy(faith_scores_, cls_scores_)

    labels = list(set(x.classification for x in annotations))
    label_to_int = {l: i for i, l in enumerate(labels)}
    key_to_instances = {inst["annotation_id"]: inst for inst in instances}
    truth = []
    predicted = []
    for ann in annotations:
        truth.append(label_to_int[ann.classification])
        inst = key_to_instances[ann.annotation_id]
        predicted.append(label_to_int[inst["classification"]])
    classification_scores = classification_report(
        truth, predicted, output_dict=True, target_names=labels, digits=3
    )
    accuracy = accuracy_score(truth, predicted)
    if "comprehensiveness_classification_scores" in instances[0]:
        comprehensiveness_scores = [
            x["classification_scores"][x["classification"]]
            - x["comprehensiveness_classification_scores"][x["classification"]]
            for x in instances
        ]
        comprehensiveness_score = np.average(comprehensiveness_scores)
    else:
        comprehensiveness_score = None
        comprehensiveness_scores = None

    if "sufficiency_classification_scores" in instances[0]:
        sufficiency_scores = [
            x["classification_scores"][x["classification"]]
            - x["sufficiency_classification_scores"][x["classification"]]
            for x in instances
        ]
        sufficiency_score = np.average(sufficiency_scores)
    else:
        sufficiency_score = None
        sufficiency_scores = None

    if "comprehensiveness_classification_scores" in instances[0]:
        comprehensiveness_entropies = [
            entropy(list(x["classification_scores"].values()))
            - entropy(list(x["comprehensiveness_classification_scores"].values()))
            for x in instances
        ]
        comprehensiveness_entropy = np.average(comprehensiveness_entropies)
        comprehensiveness_kl = np.average(
            list(
                compute_kl(
                    x["classification_scores"],
                    x["comprehensiveness_classification_scores"],
                )
                for x in instances
            )
        )
    else:
        comprehensiveness_entropies = None
        comprehensiveness_kl = None
        comprehensiveness_entropy = None

    if "sufficiency_classification_scores" in instances[0]:
        sufficiency_entropies = [
            entropy(list(x["classification_scores"].values()))
            - entropy(list(x["sufficiency_classification_scores"].values()))
            for x in instances
        ]
        sufficiency_entropy = np.average(sufficiency_entropies)
        sufficiency_kl = np.average(
            list(
                compute_kl(
                    x["classification_scores"], x["sufficiency_classification_scores"]
                )
                for x in instances
            )
        )
    else:
        sufficiency_entropies = None
        sufficiency_kl = None
        sufficiency_entropy = None

    if "thresholded_scores" in instances[0]:
        (
            aopc_thresholds,
            aopc_comprehensiveness_score,
            aopc_comprehensiveness_points,
            aopc_sufficiency_score,
            aopc_sufficiency_points,
        ) = compute_aopc_scores(instances, aopc_thresholds)
    else:
        (
            aopc_thresholds,
            aopc_comprehensiveness_score,
            aopc_comprehensiveness_points,
            aopc_sufficiency_score,
            aopc_sufficiency_points,
        ) = (None, None, None, None, None)
    if "tokens_to_flip" in instances[0]:
        token_percentages = []
        for ann in annotations:
            # in practice, this is of size 1 for everything except e-snli
            docids = set(ev.docid for ev in chain.from_iterable(ann.evidences))
            inst = key_to_instances[ann.annotation_id]
            tokens = inst["tokens_to_flip"]
            doc_lengths = sum(len(docs[d]) for d in docids)
            token_percentages.append(tokens / doc_lengths)
        token_percentages = np.average(token_percentages)
    else:
        token_percentages = None

    return {
        "accuracy": accuracy,
        "prf": classification_scores,
        "comprehensiveness": comprehensiveness_score,
        "sufficiency": sufficiency_score,
        "comprehensiveness_entropy": comprehensiveness_entropy,
        "comprehensiveness_kl": comprehensiveness_kl,
        "sufficiency_entropy": sufficiency_entropy,
        "sufficiency_kl": sufficiency_kl,
        "aopc_thresholds": aopc_thresholds,
        "comprehensiveness_aopc": aopc_comprehensiveness_score,
        "comprehensiveness_aopc_points": aopc_comprehensiveness_points,
        "sufficiency_aopc": aopc_sufficiency_score,
        "sufficiency_aopc_points": aopc_sufficiency_points,
    }
