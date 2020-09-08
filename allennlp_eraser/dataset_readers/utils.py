from dataclasses import dataclass
from itertools import chain
from typing import FrozenSet, Set, Tuple, Union


@dataclass(eq=True, frozen=True)
class Evidence:
    """
    (docid, start_token, end_token) form the only official Evidence; sentence level annotations are for convenience.
    Args:
        text: Some representation of the evidence text
        docid: Some identifier for the document
        start_token: The canonical start token, inclusive
        end_token: The canonical end token, exclusive
        start_sentence: Best guess start sentence, inclusive
        end_sentence: Best guess end sentence, exclusive
    """

    text: Union[str, Tuple[int], Tuple[str]]
    docid: str
    start_token: int = -1
    end_token: int = -1
    start_sentence: int = -1
    end_sentence: int = -1


@dataclass(eq=True, frozen=True)
class Annotation:
    """
    Args:
        annotation_id: unique ID for this annotation element
        query: some representation of a query string
        evidences: a set of "evidence groups".
            Each evidence group is:
                * sufficient to respond to the query (or justify an answer)
                * composed of one or more Evidences
                * may have multiple documents in it (depending on the dataset)
                    - e-snli has multiple documents
                    - other datasets do not
        classification: str
        query_type: Optional str, additional information about the query
        docids: a set of docids in which one may find evidence.
    """

    annotation_id: str
    query: Union[str, Tuple[int]]
    evidences: Union[Set[Tuple[Evidence]], FrozenSet[Tuple[Evidence]]]
    classification: str
    query_type: str = None
    docids: Set[str] = None

    def all_evidences(self) -> Tuple[Evidence]:
        return tuple(list(chain.from_iterable(self.evidences)))


def check_phase(phase: str) -> None:
    if phase not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid phase: {phase}")
