import pytest
from allennlp.common.util import ensure_list

from allennlp_eraser.common.testing import AllenNlpEraserTestCase
from allennlp_eraser.dataset_readers import BoolqDatasetReader, BoolqEraserDatasetReader


class TestBoolqEraserDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_lazy(self, lazy: bool):
        reader = BoolqEraserDatasetReader(lazy=lazy)
        instances = ensure_list(reader.read("train.jsonl"))

        assert len(instances) == 6363

    @pytest.mark.parametrize(
        "file_path, num_data",
        (
            ("train.jsonl", 6363),
            ("train_data.jsonl", 6363),
            ("val.jsonl", 1491),
            ("dev_data.jsonl", 1491),
            ("test.jsonl", 2807),
            ("test_original.jsonl", 2817),
            ("test_comprehensive.jsonl", 199),
            ("test_data.jsonl", 2817),
        ),
    )
    def test_read_from_file(self, file_path, num_data):
        reader = BoolqEraserDatasetReader(lazy=True)
        instances = ensure_list(reader.read(file_path))
        assert len(instances) == num_data


class TestBoolqDatasetReader:
    @property
    def instance1(self):
        passage = [
            "Good",
            "Samaritan",
            "laws",
            "offer",
            "legal",
            "protection",
            "to",
            "people",
            "who",
            "give",
            "reasonable",
            "assistance",
            "to",
            "those",
            "who",
            "are",
            ",",
            "or",
            "who",
            "they",
            "believe",
            "to",
            "be",
            ",",
            "injured",
            ",",
            "ill",
            ",",
            "in",
            "peril",
            ",",
            "or",
            "otherwise",
            "incapacitated",
            ".",
            "The",
            "protection",
            "is",
            "intended",
            "to",
            "reduce",
            "bystanders",
            "'",
            "hesitation",
            "to",
            "assist",
            ",",
            "for",
            "fear",
            "of",
            "being",
            "sued",
            "or",
            "prosecuted",
            "for",
            "unintentional",
            "injury",
            "or",
            "wrongful",
            "death",
            ".",
            "An",
            "example",
            "of",
            "such",
            "a",
            "law",
            "in",
            "common",
            "-",
            "law",
            "areas",
            "of",
            "Canada",
            ":",
            "a",
            "good",
            "Samaritan",
            "doctrine",
            "is",
            "a",
            "legal",
            "principle",
            "that",
            "prevents",
            "a",
            "rescuer",
            "who",
            "has",
            "voluntarily",
            "helped",
            "a",
            "victim",
            "in",
            "distress",
            "from",
            "being",
            "successfully",
            "sued",
            "for",
            "wrongdoing",
            ".",
            "Its",
            "purpose",
            "is",
            "to",
            "keep",
            "people",
            "from",
            "being",
            "reluctant",
            "to",
            "help",
            "a",
            "stranger",
            "in",
            "need",
            "for",
            "fear",
            "of",
            "legal",
            "repercussions",
            "should",
            "they",
            "make",
            "some",
            "mistake",
            "in",
            "treatment",
            ".",
            "By",
            "contrast",
            ",",
            "a",
            "duty",
            "to",
            "rescue",
            "law",
            "requires",
            "people",
            "to",
            "offer",
            "assistance",
            "and",
            "holds",
            "those",
            "who",
            "fail",
            "to",
            "do",
            "so",
            "liable",
            ".",
        ]
        title = ["Good", "Samaritan", "law"]
        question = [
            "do",
            "good",
            "samaritan",
            "laws",
            "protect",
            "those",
            "who",
            "help",
            "at",
            "an",
            "accident",
        ]
        answer = True
        return {
            "passage": passage,
            "title": title,
            "question": question,
            "answer": answer,
        }

    @property
    def instance2(self):

        passage = [
            "Windows",
            "Movie",
            "Maker",
            "(",
            "formerly",
            "known",
            "as",
            "Windows",
            "Live",
            "Movie",
            "Maker",
            "in",
            "Windows",
            "7",
            ")",
            "is",
            "a",
            "discontinued",
            "video",
            "editing",
            "software",
            "by",
            "Microsoft",
            ".",
            "It",
            "is",
            "a",
            "part",
            "of",
            "Windows",
            "Essentials",
            "software",
            "suite",
            "and",
            "offers",
            "the",
            "ability",
            "to",
            "create",
            "and",
            "edit",
            "videos",
            "as",
            "well",
            "as",
            "to",
            "publish",
            "them",
            "on",
            "OneDrive",
            ",",
            "Facebook",
            ",",
            "Vimeo",
            ",",
            "YouTube",
            ",",
            "and",
            "Flickr",
            ".",
        ]
        title = ["Windows", "Movie", "Maker"]
        question = [
            "is",
            "windows",
            "movie",
            "maker",
            "part",
            "of",
            "windows",
            "essentials",
        ]
        answer = True

        return {
            "passage": passage,
            "title": title,
            "question": question,
            "answer": answer,
        }

    @property
    def instance3(self):
        passage = [
            "Powdered",
            "sugar",
            ",",
            "also",
            "called",
            "confectioners",
            "'",
            "sugar",
            ",",
            "icing",
            "sugar",
            ",",
            "and",
            "icing",
            "cake",
            ",",
            "is",
            "a",
            "finely",
            "ground",
            "sugar",
            "produced",
            "by",
            "milling",
            "granulated",
            "sugar",
            "into",
            "a",
            "powdered",
            "state",
            ".",
            "It",
            "usually",
            "contains",
            "a",
            "small",
            "amount",
            "of",
            "anti",
            "-",
            "caking",
            "agent",
            "to",
            "prevent",
            "clumping",
            "and",
            "improve",
            "flow",
            ".",
            "Although",
            "most",
            "often",
            "produced",
            "in",
            "a",
            "factory",
            ",",
            "powdered",
            "sugar",
            "can",
            "also",
            "be",
            "made",
            "by",
            "processing",
            "ordinary",
            "granulated",
            "sugar",
            "in",
            "a",
            "coffee",
            "grinder",
            ",",
            "or",
            "by",
            "crushing",
            "it",
            "by",
            "hand",
            "in",
            "a",
            "mortar",
            "and",
            "pestle",
            ".",
        ]
        title = ["Powdered", "sugar"]
        question = [
            "is",
            "confectionary",
            "sugar",
            "the",
            "same",
            "as",
            "powdered",
            "sugar",
        ]
        answer = True

        return {
            "passage": passage,
            "title": title,
            "question": question,
            "answer": answer,
        }

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_boolq(self, lazy: bool):
        reader = BoolqDatasetReader(lazy=lazy)
        file_path = AllenNlpEraserTestCase.FIXTURES_ROOT / "data" / "boolq.jsonl"
        instances = reader.read(file_path)
        instances = ensure_list(instances)

        assert len(instances) == 3

        fields = instances[0].fields
        instance1 = self.instance1
        for key in instance1.keys():
            if key != "answer":
                assert [t.text for t in fields[key].tokens] == instance1[key]
            else:
                assert bool(fields[key].label) == instance1[key]

        fields = instances[1].fields
        instance2 = self.instance2
        for key in instance2.keys():
            if key != "answer":
                assert [t.text for t in fields[key].tokens] == instance2[key]
            else:
                assert bool(fields[key].label) == instance2[key]

        fields = instances[2].fields
        instance3 = self.instance3
        for key in instance3.keys():
            if key != "answer":
                assert [t.text for t in fields[key].tokens] == instance3[key]
            else:
                assert bool(fields[key].label) == instance3[key]
