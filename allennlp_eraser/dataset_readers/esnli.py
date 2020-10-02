import json
from typing import Dict, Iterable, List, Optional

from allennlp.common import JsonDict
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Token, Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from overrides import overrides


@DatasetReader.register("esnli")
class ESNLIDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        segment_sentences: bool = False,
        max_sequence_length: Optional[int] = None,
        skip_label_indexing: bool = False,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multi_process_sharding: bool = False,
    ):
        super().__init__(
            lazy=lazy,
            cache_directory=cache_directory,
            max_instances=max_instances,
            manual_distributed_sharding=manual_distributed_sharding,
            manual_multi_process_sharding=manual_multi_process_sharding,
        )

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing

        if segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    def _truncate_tokens(self, tokens: List[Token]) -> List[Token]:
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:, self._max_sequence_length]
        return tokens

    def _truncate(self, tokens: List[Token]) -> List[Token]:
        if self._max_sequence_length:
            tokens = self._truncate_tokens(tokens)
        return tokens

    def cleanup_data(self, data: JsonDict) -> JsonDict:

        data["pair_id"] = data.pop("pairID")
        data["label"] = data.pop("gold_label")
        data["premise"] = data.pop("Sentence1")
        data["hypothesis"] = data.pop("Sentence2")
        data["premise_marked"] = data.pop("Sentence1_marked_1")
        data["hypothesis_marked"] = data.pop("Sentence2_marked_1")
        data["explanation"] = data.pop("Explanation_1")

        del data["WorkerID"]
        del data["Sentence1_Highlighted_1"]
        del data["Sentence2_Highlighted_1"]

        return data

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as rf:
            for line in rf:
                data = json.loads(line)
                data = self.cleanup_data(data)

                yield self.text_to_instance(**data)

    @overrides
    def text_to_instance(
        self,
        premise: str,
        premise_marked: str,
        hypothesis: str,
        hypothesis_marked: str,
        label: Optional[str] = None,
        pair_id: Optional[str] = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(premise)
            for sentence in sentence_splits:
                tokens = self._tokenizer.tokenize(sentence)
                tokens = self._truncate(tokens)
                sentences.append(TextField(tokens, self._token_indexers))
            fields["premise"] = ListField(sentences)

        else:
            tokens = self._tokenizer.tokenize(premise)
            tokens = self._truncate(tokens)
            fields["premise"] = TextField(tokens, self._token_indexers)

        fields["hypothesis"] = TextField(
            self._tokenizer.tokenize(hypothesis), self._token_indexers
        )
        fields["metadata"] = MetadataField({"pair_id": pair_id})

        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=self._skip_label_indexing)

        return Instance(fields)
