from itertools import chain
from typing import List, Tuple

import numpy as np


def _instances_aopc(
    instances: List[dict], thresholds: List[float], key: str
) -> Tuple[float, List[float]]:
    dataset_scores = []
    for inst in instances:
        kls = inst["classification"]
        beta_0 = inst["classification_scores"][kls]
        instance_scores = []
        for score in filter(
            lambda x: x["threshold"] in thresholds,
            sorted(inst["thresholded_scores"], key=lambda x: x["threshold"]),
        ):
            beta_k = score[key][kls]
            delta = beta_0 - beta_k
            instance_scores.append(delta)
        assert len(instance_scores) == len(thresholds)
        dataset_scores.append(instance_scores)
    dataset_scores = np.array(dataset_scores)
    # a careful reading of Samek, et al. "Evaluating the Visualization of What a Deep Neural Network Has Learned"
    # and some algebra will show the reader that we can average in any of several ways and get the same result:
    # over a flattened array, within an instance and then between instances, or over instances (by position) an
    # then across them.
    final_score = np.average(dataset_scores)
    position_scores = np.average(dataset_scores, axis=0).tolist()

    return final_score, position_scores


def compute_aopc_scores(instances: List[dict], aopc_thresholds: List[float]):
    if aopc_thresholds is None:
        aopc_thresholds = sorted(
            set(
                chain.from_iterable(
                    [x["threshold"] for x in y["thresholded_scores"]] for y in instances
                )
            )
        )
    aopc_comprehensiveness_score, aopc_comprehensiveness_points = _instances_aopc(
        instances, aopc_thresholds, "comprehensiveness_classification_scores"
    )
    aopc_sufficiency_score, aopc_sufficiency_points = _instances_aopc(
        instances, aopc_thresholds, "sufficiency_classification_scores"
    )
    return (
        aopc_thresholds,
        aopc_comprehensiveness_score,
        aopc_comprehensiveness_points,
        aopc_sufficiency_score,
        aopc_sufficiency_points,
    )
