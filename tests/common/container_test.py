from dataclasses import asdict

from allennlp_eraser.common.container import (
    InstanceResult,
    RationaleResult,
    SpanResult,
    ThresholdedScore,
)


class TestSpanResult:
    def test_span_result(self):
        span = SpanResult(start_token=1, end_token=3)
        assert asdict(span) == {"start_token": 1, "end_token": 3}


class TestRationaleResult:
    def test_rational_result(self):
        docid = "00001"
        hard_rationale_pred = [SpanResult(0, 1), SpanResult(2, 3)]
        soft_rationale_pred = [SpanResult(3, 4), SpanResult(6, 9)]
        soft_sentence_pred = [SpanResult(2, 5), SpanResult(8, 11)]
        rationale = RationaleResult(
            docid=docid,
            hard_rationale_predictions=hard_rationale_pred,
            soft_rationale_predictions=soft_rationale_pred,
            soft_sentence_predictions=soft_sentence_pred,
        )
        assert asdict(rationale) == {
            "docid": "00001",
            "hard_rationale_predictions": [
                {"start_token": 0, "end_token": 1},
                {"start_token": 2, "end_token": 3},
            ],
            "soft_rationale_predictions": [
                {"start_token": 3, "end_token": 4},
                {"start_token": 6, "end_token": 9},
            ],
            "soft_sentence_predictions": [
                {"start_token": 2, "end_token": 5},
                {"start_token": 8, "end_token": 11},
            ],
        }

    def test_empty_rational_result(self):

        hard_spans = [SpanResult(0, 1), SpanResult(2, 4)]
        rationale = RationaleResult(docid="00001", hard_rationale_prediction=hard_spans)
        assert len(rationale.hard_rationale_predictions) == 2
        assert len(rationale.soft_rationale_predictions) == 0
        assert len(rationale.soft_sentence_predictions) == 0


class TestThresholdedScore:
    def test_thresholded_score(self):
        score = ThresholdedScore()


class TestInstanceResult:
    def test_instance_result(self):
        score = InstanceResult()
