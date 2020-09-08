import json
import pathlib
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
from zipfile import ZipFile, ZipInfo

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Token, Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from overrides import overrides

from allennlp_eraser.dataset_readers import EraserDatasetReader
from allennlp_eraser.dataset_readers.utils import Annotation, Evidence, check_phase


@DatasetReader.register("movies")
class MoviesDatasetReader(DatasetReader):
    def __init__(
        self,
        dataset_url: str = "http://www.cs.jhu.edu/~ozaidan/rationales/review_polarity_rationales.zip",
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[TokenIndexer] = None,
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

    def _is_file_in_phase(self, phase: str, num: int) -> bool:
        if phase == "train":
            return True if num >= 0 and 800 > num else False
        elif phase == "valid":
            return True if num >= 800 and 900 > num else False
        elif phase == "test":
            return True if num >= 900 and 1000 > num else False

    def _get_num_from_filename(self, filename: str) -> int:
        return int(filename.split("_")[-1])

    def _find_file_from_zip(
        self, phase: str, zip_file: ZipFile
    ) -> Iterable[Tuple[ZipInfo, str]]:

        zip_info_list = zip_file.infolist()
        for zip_info in zip_info_list:
            zip_name = pathlib.Path(zip_info.filename)

            label = None
            if zip_name.match("noRats_pos/*"):
                label = "positive"
            elif zip_name.match("noRats_neg/*"):
                label = "negative"

            if label is not None:
                num = self._get_num_from_filename(zip_name.stem)
                if self._is_file_in_phase(phase, num):
                    yield zip_info, label

    def _read_zipfile(self, phase: str):
        file_path = cached_path(self._dataset_url)

        with ZipFile(file_path, "r") as zip_file:
            for zip_info, label in self._find_file_from_zip(phase, zip_file):
                with zip_file.open(zip_info, "r") as rf:
                    line = rf.read().decode("utf-8").rstrip()
                    yield line, label

    @overrides
    def _read(self, phase: str) -> Iterable[Instance]:
        check_phase(phase)
        for text, label in self._read_zipfile(phase):
            yield self.text_to_instance(text, label)

    def _truncate_tokens(self, tokens: List[Token]) -> List[Token]:
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:, self._max_sequence_length]
        return tokens

    def _truncate(self, tokens: List[Token]) -> List[Token]:
        if self._max_sequence_length:
            tokens = self._truncate_tokens(tokens)
        return tokens

    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                tokens = self._tokenizer.tokenize(sentence)
                tokens = self._truncate(tokens)
                sentences.append(TextField(tokens, self._token_indexers))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens, self._token_indexers)

        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)


@DatasetReader.register("eraser-movies")
class MoviesEraserDatasetReader(EraserDatasetReader):
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
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

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        for line in self._read_tarfile(file_path, dataset_name="movies"):
            data = json.loads(line)
            yield self.text_to_instance(**data)
