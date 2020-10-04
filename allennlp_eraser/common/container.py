from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class SpanResult(object):
    start_token: str
    end_token: str


@dataclass
class RationaleResult(object):
    docid: str
    hard_rationale_predictions: List[SpanResult] = field(default_factory=list)
    soft_rationale_predictions: List[SpanResult] = field(default_factory=list)
    soft_sentence_predictions: List[SpanResult] = field(default_factory=list)


@dataclass
class ThresholdedScore(object):
    threshold: float
    comprehensiveness_classification_scores: Any
    sufficiency_classification_scores: Any


@dataclass
class InstanceResult(object):
    annotation_id: str
    rationales: List[RationaleResult]
    classification: str
    classification_scores: dict
    comprehensiveness_classification_scores: dict
    sufficiency_classification_scores: dict
    thresholded_scores: List[ThresholdedScore]
