import pathlib
import tarfile
from tarfile import TarFile, TarInfo
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, MetadataField, TextField
from allennlp.data.instance import Instance
from overrides import overrides

from allennlp_eraser.dataset_readers.utils import Annotation, Evidence


class EraserDatasetReader(DatasetReader):
    def __init__(
        self,
        eraser_dataset_path: str = "https://storage.googleapis.com/sfr-nazneen-website-files-research/data_v1.2.tar.gz",
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
        self._eraser_dataset_path = eraser_dataset_path

    def _find_file_from_tar(
        self,
        tar: TarFile,
        filename: str,
        dataset_name: str,
    ) -> TarInfo:
        for tarinfo in tar:
            tarname = pathlib.Path(tarinfo.name)
            if tarname.match(f"data/{dataset_name}/{filename}"):
                return tarinfo

        raise FileNotFoundError(filename)

    def _read_tarfile(self, filename: str, dataset_name: str) -> TarInfo:
        file_path = cached_path(self._eraser_dataset_path)

        with tarfile.open(file_path, "r") as tar_file:
            tar_info = self._find_file_from_tar(tar_file, filename, dataset_name)

            for line in tar_file.extractfile(tar_info):
                yield line

    def get_evidences(
        self, evidences: Union[Set[Tuple[Evidence]], FrozenSet[Tuple[Evidence]]]
    ) -> List[Evidence]:
        es = []
        for evidence in evidences:
            if isinstance(evidence, list):
                for e in evidence:
                    es.append(Evidence(**e))
            elif isinstance(evidence, dict):
                es.append(Evidence(**evidence))
            else:
                breakpoint()
        return es

    @overrides
    def text_to_instance(
        self,
        annotation_id: int,
        classification: str,
        evidences: Union[Set[Tuple[Evidence]], FrozenSet[Tuple[Evidence]]],
        query: Union[str, Tuple[int]],
        docids: Optional[Set[str]] = None,
        query_type: Optional[str] = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}

        query_tokenized = self._tokenizer.tokenize(query)
        fields["query"] = TextField(query_tokenized, self._token_indexers)

        evidences = self.get_evidences(evidences)
        annotation = Annotation(
            annotation_id=annotation_id,
            query=query,
            evidences=evidences,
            classification=classification,
            query_type=query_type,
            docids=docids,
        )
        metadata = {"annotation": annotation}
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
