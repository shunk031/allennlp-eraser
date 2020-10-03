import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
from allennlp_eraser.common.util import (
    Annotation,
    Evidence,
    annotations_from_jsonl,
    load_documents,
    load_flattened_documents,
    load_jsonl,
)
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _has_hard_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return (
        "rationales" in results[0]
        and len(results[0]["rationales"]) > 0
        and "hard_rationale_predictions" in results[0]["rationales"][0]
        and results[0]["rationales"][0]["hard_rationale_predictions"] is not None
        and len(results[0]["rationales"][0]["hard_rationale_predictions"]) > 0
    )


def _has_soft_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return (
        "rationales" in results[0]
        and len(results[0]["rationales"]) > 0
        and "soft_rationale_predictions" in results[0]["rationales"][0]
        and results[0]["rationales"][0]["soft_rationale_predictions"] is not None
    )


def _has_soft_sentence_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return (
        "rationales" in results[0]
        and len(results[0]["rationales"]) > 0
        and "soft_sentence_predictions" in results[0]["rationales"][0]
        and results[0]["rationales"][0]["soft_sentence_predictions"] is not None
    )


def _has_classifications(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return "classification" in results[0] and results[0]["classification"] is not None
