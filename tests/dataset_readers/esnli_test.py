import pytest
from allennlp.common.util import ensure_list

from allennlp_eraser.common.testing import AllenNlpEraserTestCase
from allennlp_eraser.dataset_readers import EraserDatasetReader

# from allennlp_eraser.dataset_readers import ESNLIDatasetReader


class TestESNLIEraserDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_lazy(self, lazy: bool):
        reader = EraserDatasetReader(lazy=lazy)
        file_path = (
            AllenNlpEraserTestCase.FIXTURES_ROOT / "data" / "esnli" / "train.jsonl"
        )
        instances1 = ensure_list(reader.read(file_path))
        file_path = (
            AllenNlpEraserTestCase.FIXTURES_ROOT / "data" / "esnli_flat" / "train.jsonl"
        )
        instances2 = ensure_list(reader.read(file_path))
        assert (len(instances1) + len(instances2)) == 911938
