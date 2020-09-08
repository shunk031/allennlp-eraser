import json

import pytest

from allennlp_eraser.dataset_readers import EraserDatasetReader


class TestEraserDatasetReader:
    def read_from_file(self, file_path, dataset_name):
        reader = EraserDatasetReader()

        line = next(reader._read_tarfile(file_path, dataset_name))
        data = json.loads(line)
        assert isinstance(data, dict)

    @pytest.mark.parametrize(
        "dataset_name, file_path",
        [
            ("boolq", "train.jsonl"),
            ("boolq", "train_data.jsonl"),
            ("boolq", "val.jsonl"),
            ("boolq", "dev_data.jsonl"),
            ("boolq", "test.jsonl"),
            ("boolq", "test_data.jsonl"),
            ("boolq", "test_original.jsonl"),
            ("boolq", "test_comprehensive.jsonl"),
        ],
    )
    def test_read_boolq(self, dataset_name, file_path):
        self.read_from_file(dataset_name, file_path)

    @pytest.mark.parametrize(
        "dataset_name, file_path",
        [("cose", "train.jsonl"), ("cose", "train_sanity.jsonl")],
    )
    def test_read_cose(self, dataset_name, file_path):
        self.read_from_file(dataset_name, file_path)
