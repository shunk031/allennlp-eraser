import pathlib
import tarfile
from tarfile import TarFile, TarInfo
from typing import Iterator, Optional

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader


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

        with tarfile.open(file_path, "r") as tar:
            tarinfo = self._find_file_from_tar(tar, filename, dataset_name)

            for line in tar.extractfile(tarinfo):
                yield line
