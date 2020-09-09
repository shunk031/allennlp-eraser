import pytest
from allennlp.common.util import ensure_list

from allennlp_eraser.common.testing import AllenNlpEraserTestCase
from allennlp_eraser.dataset_readers import BoolqDatasetReader, EraserDatasetReader


class TestMultiREraserDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_lazy(self, lazy: bool):
        reader = EraserDatasetReader(lazy=lazy)
        file_path = (
            AllenNlpEraserTestCase.FIXTURES_ROOT / "data" / "multirc" / "train.jsonl"
        )
        instances = ensure_list(reader.read(file_path))
        assert len(instances) == 24029

    @pytest.mark.parametrize(
        "filename, num_data",
        (
            ("train.jsonl", 24029),
            ("val.jsonl", 3214),
            ("test.jsonl", 4848),
        ),
    )
    def test_read_from_file(self, filename: str, num_data: int):
        reader = EraserDatasetReader(lazy=True)
        file_path = AllenNlpEraserTestCase.FIXTURES_ROOT / "data" / "multirc" / filename
        instances = ensure_list(reader.read(file_path))
        assert len(instances) == num_data
