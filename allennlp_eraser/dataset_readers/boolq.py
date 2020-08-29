import json
from typing import Iterable, List, Optional

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from overrides import overrides

from allennlp_eraser.dataset_readers import EraserDatasetReader


@DatasetReader.register("boolq")
class BoolqDatasetReader(EraserDatasetReader):
    def __init__(
        self,
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

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        for line in self._read_tarfile(file_path, dataset_name="boolq"):
            data = json.loads(line)
            yield self.text_to_instance(**data)

    @overrides
    def text_to_instance(
        self,
        annotation_id: int,
        classification: bool,
        docids: List[str],
        evidences,
        query,
        query_type,
    ) -> Instance:
        return annotation_id
