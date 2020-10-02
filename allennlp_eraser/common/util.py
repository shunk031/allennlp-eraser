import json
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Dict, FrozenSet, List, Set, Tuple, Union


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


def load_jsonl(file_path: str) -> List[dict]:
    ret = []
    with open(file_path, "r") as rf:
        for line in rf:
            content = json.loads(line)
            ret.append(content)
    return ret


def annotations_from_jsonl(file_path: str) -> List[Annotation]:

    ret = []
    with open(file_path, "r") as rf:
        for line in rf:
            content = json.loads(line)

            ev_groups = []
            for ev_group in content["evidences"]:
                ev_group = tuple([Evidence(**ev) for ev in ev_group])
                ev_groups.append(ev_group)
            content["evidences"] = frozenset(ev_groups)
            ret.append(Annotation(**content))

    return ret


def load_documents(
    data_dir: str, docids: Set[str] = None
) -> Dict[str, List[List[str]]]:
    """Loads a subset of available documents from disk.
    Each document is assumed to be serialized as newline ('\n') separated sentences.
    Each sentence is assumed to be space (' ') joined tokens.
    """
    if os.path.exists(os.path.join(data_dir, "docs.jsonl")):
        assert not os.path.exists(os.path.join(data_dir, "docs"))
        return load_documents_from_file(data_dir, docids)

    docs_dir = os.path.join(data_dir, "docs")
    res = dict()
    if docids is None:
        docids = sorted(os.listdir(docs_dir))
    else:
        docids = sorted(set(str(d) for d in docids))
    for d in docids:
        with open(os.path.join(docs_dir, d), "r") as rf:
            lines = [line.strip() for line in rf.readlines()]
            lines = list(filter(lambda x: bool(len(x)), lines))
            # tokenized = [
            #     list(filter(lambda x: bool(len(x)), line.strip().split(" ")))
            #     for line in lines
            # ]
            res[d] = lines
    return res


def load_flattened_documents(data_dir: str, docids: Set[str]) -> Dict[str, List[str]]:
    """Loads a subset of available documents from disk.
    Returns a tokenized version of the document.
    """
    unflattened_docs = load_documents(data_dir, docids)
    flattened_docs = {}
    for doc, unflattened in unflattened_docs.items():
        flattened_docs[doc] = list(chain.from_iterable(unflattened))
    return flattened_docs


def load_documents_from_file(
    data_dir: str, docids: Set[str] = None
) -> Dict[str, List[List[str]]]:
    """Loads a subset of available documents from 'docs.jsonl' file on disk.
    Each document is assumed to be serialized as newline ('\n') separated sentences.
    Each sentence is assumed to be space (' ') joined tokens.
    """
    docs_file = os.path.join(data_dir, "docs.jsonl")
    documents = load_jsonl(docs_file)
    documents = {doc["docid"]: doc["document"] for doc in documents}
    res = dict()
    if docids is None:
        docids = sorted(list(documents.keys()))
    else:
        docids = sorted(set(str(d) for d in docids))
    for d in docids:
        lines = documents[d].split("\n")
        # tokenized = [line.strip().split(" ") for line in lines]
        res[d] = lines
    return res


def sort_docids_from_evidences(evidences: List[List[Evidence]]) -> List[str]:
    unique_ids = set(
        [ev_clause.docid for ev_group in evidences for ev_clause in ev_group]
    )
    return sorted(list(unique_ids))


def generate_doc_evidence_map(
    self, evidences: List[List[Evidence]]
) -> Dict[str, List[Tuple[int, int]]]:

    doc_evidence_map: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for ev_group in evidences:
        for ev_clause in ev_group:
            doc_evidence_map[ev_clause.docid].append(
                (ev_clause.start_token, ev_clause.end_token)
            )
    return doc_evidence_map
