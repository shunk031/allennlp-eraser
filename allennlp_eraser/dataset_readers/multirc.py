import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from zipfile import ZipFile

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from overrides import overrides

from allennlp_eraser.common.util import check_phase

DATASET_URL = "https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip"


@dataclass(eq=True, frozen=True)
class MultiRCAnswer(object):
    text: str
    isAnswer: bool
    scores: List[int]


@DatasetReader.register("multirc")
class MultiRCDatasetReader(DatasetReader):
    def __init__(
        self,
        dataset_url: str = DATASET_URL,
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
        self._dataset_url = dataset_url
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing

        if segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    def get_filename_in_zip(self, phase: str) -> str:
        if phase == "train":
            return "splitv2/train_456-fixedIds.json"

        else:
            return "splitv2/dev_83-fixedIds.json"

    @overrides
    def _read(self, phase: str) -> Iterable[Instance]:
        check_phase(phase)
        file_path = cached_path(self._dataset_url)
        with ZipFile(file_path, "r") as zip_file:
            with zip_file.open(self._get_filename_in_zip(phase), "r") as rf:
                content = json.load(rf)

        for data in content["data"]:
            paragraph = data["paragraph"]
            text = paragraph["text"]
            questions = paragraph["questions"]

            for question in questions:
                answers = [MultiRCAnswer(**ans) for ans in question.pop("answers")]
                for answer in answers:
                    yield self.text_to_instance(text=text, answer=answer, **question)

    @overrides
    def text_to_instance(
        self,
        text: str,
        question: str,
        idx: str,
        multisent: bool,
        sentence_used: List[int],
        answer: Optional[MultiRCAnswer] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                tokens = self._tokenizer.tokenize(sentence)
                tokens = self._truncate(tokens)
                sentences.append(TextField(tokens, self._token_indexers))
            fields["passage"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            tokens = self._truncate(tokens)
            fields["passage"] = TextField(tokens, self._token_indexers)

        fields["question"] = TextField(
            self._tokenizer.tokenize(question), self._token_indexers
        )

        metadata = {"idx": idx, "multisent": multisent, "sentence_used": sentence_used}
        fields["metadata"] = MetadataField(metadata)

        if answer is not None:
            fields["answer_tokens"] = TextField(
                self._tokenizer.tokenize(answer["text"]), self._token_indexers
            )
            fields["label"] = LabelField(str(answer["isAnswer"]))

        return Instance(fields)
