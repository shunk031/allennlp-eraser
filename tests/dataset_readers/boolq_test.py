import pytest
from allennlp.common.util import ensure_list

from allennlp_eraser.dataset_readers import BoolqDatasetReader


class TestBoolqReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_lazy(self, lazy):
        reader = BoolqDatasetReader(lazy=lazy)
        instances = ensure_list(reader.read("train.jsonl"))

        assert len(instances) == 6363

    @pytest.mark.parametrize(
        "file_path",
        (
            "train.jsonl",
            "train_data.jsonl",
            "val.jsonl",
            "dev_data.jsonl",
            "test.jsonl",
            "test_original.jsonl",
            "test_comprehensive.jsonl",
            "test_data.jsonl",
        ),
    )
    def test_read_from_file(self, file_path):
        reader = BoolqDatasetReader(lazy=True)
        instances = ensure_list(reader.read(file_path))
