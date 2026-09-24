from autotraining.evaluation.quality_gate import (
    evaluate_quality_gate,
)


def test_quality_gate_passes():

    result = evaluate_quality_gate(
        candidate_metric=11.5,
        incumbent_metric=11.378989,
        max_degradation=0.05,
    )

    assert result["passed"] is True


def test_quality_gate_rejects():

    result = evaluate_quality_gate(
        candidate_metric=13.0,
        incumbent_metric=11.378989,
        max_degradation=0.05,
    )

    assert result["passed"] is False