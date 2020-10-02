import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from allennlp.data import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import (
    Field,
    LabelField,
    MetadataField,
    SequenceLabelField,
    TextField
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from allennlp_eraser.common.util import (
    Annotation,
    Evidence,
    annotations_from_jsonl,
    generate_doc_evidence_map,
    load_flattened_documents,
    sort_docids_from_evidences
)
from overrides import overrides

ERASER_DATASET_URL = (
    "https://storage.googleapis.com/sfr-nazneen-website-files-research/data_v1.2.tar.gz"
)


class EraserDatasetReader(DatasetReader):
    SEP = "[SEP]"

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        max_sequence_length: Optional[int] = None,
        keep_prob: float = 1.0,
        evidence_labels_namespace: str = "evidence_labels",
        kept_token_labels_namespace: str = "kept_token_labels",
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multi_process_sharding: bool = False,
    ) -> None:

        super().__init__(
            lazy=lazy,
            cache_directory=cache_directory,
            max_instances=max_instances,
            manual_distributed_sharding=manual_distributed_sharding,
            manual_multi_process_sharding=manual_multi_process_sharding,
        )

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._is_bert = "bert" in self._token_indexers

        self._max_sequence_length = max_sequence_length
        self._keep_prob = keep_prob

        self._evidence_labels_namespace = evidence_labels_namespace
        self._kept_token_labels_namespace = kept_token_labels_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        data_dir = os.path.dirname(file_path)
        annotations: List[Annotation] = annotations_from_jsonl(file_path)
        docs: Dict[str, List[str]] = load_flattened_documents(data_dir, docids=None)

        for ann in annotations:
            annotation_id: str = ann.annotation_id
            evidences: List[List[Evidence]] = ann.evidences
            label: str = ann.classification
            query: str = ann.query
            docids: List[str] = sort_docids_from_evidences(evidences)

            filtered_docs: Dict[str, List[str]] = {d: docs[d] for d in docids}
            doc_evidence_map = generate_doc_evidence_map(evidences)

            if label is not None:
                label = str(label)

            yield self.text_to_instance(
                annotation_id=annotation_id,
                docs=filtered_docs,
                rationales=doc_evidence_map,
                query=query,
                label=label,
            )

    @overrides
    def text_to_instance(
        self,
        annotation_id: str,
        docs: Dict[str, List[str]],
        rationales: Dict[str, List[Tuple[int, int]]],
        query: str = None,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}

        tokens: List[Token] = []
        is_evidence = []
        doc_to_span_map: Dict[str, Tuple[int, int]] = {}
        always_keep_mask: List[int] = []

        for docid, doc_words in docs.items():
            # doc_tokens = [Token(w) for w in doc_words]
            doc_tokens = self._tokenizer.tokenize(doc_words)
            tokens.extend(doc_tokens)
            doc_to_span_map[docid] = (len(tokens) - len(doc_words), len(tokens))

            always_keep_mask.extend([0] * len(doc_tokens))
            tokens.append(Token(self.SEP))
            always_keep_mask.extend([1])

            rationale: List[int] = [0] * len(doc_words)
            if docid in rationales:
                for start_token, end_token in rationales[docid]:
                    for i in range(start_token, end_token):
                        rationale[i] = 1
            is_evidence.extend(rationale + [1])

        if (query is not None) and (not isinstance(query, list)):
            query_words = query.split()
            tokens.extend([Token(w) for w in query_words])
            tokens.append(Token(self.SEP))
            is_evidence.extend([1] * (len(query_words) + 1))
            always_keep_mask.extend([1] * (len(query_words) + 1))

        fields["doc"] = TextField(tokens, self._token_indexers)
        fields["rationale"] = SequenceLabelField(
            is_evidence,
            sequence_field=fields["doc"],
            label_namespace=self._evidence_labels_namespace,
        )
        fields["kept_tokens"] = SequenceLabelField(
            always_keep_mask,
            sequence_field=fields["doc"],
            label_namespace=self._kept_token_labels_namespace,
        )

        metadata = {
            "annotation_id": annotation_id,
            "tokens": tokens,
            "doc_to_span_map": doc_to_span_map,
            "convert_tokens_to_instance": self._convert_tokens_to_instances,
            "always_keep_mask": np.array(always_keep_mask),
        }
        fields["metadata"] = MetadataField(metadata)

        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)

    def _convert_tokens_to_instances(
        self, tokens: List[Token], labels: str = None
    ) -> List[Instance]:
        return [Instance({"doc": TextField(tokens, self._token_indexers)})]
