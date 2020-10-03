from dataclasses import dataclass
from typing import Any


@dataclass
class RationalResult(object):
    docid: str
    hard_rational_predictions: Any
    soft_rationale_predictions: Any
    soft_sentence_predictions: Any
