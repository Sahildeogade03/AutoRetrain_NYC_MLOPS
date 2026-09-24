"""Module scaffold: quality_gate."""

from __future__ import annotations


def evaluate_quality_gate(
    candidate_metric: float,
    incumbent_metric: float,
    max_degradation: float,
) -> dict:

    allowed_metric = (
        incumbent_metric
        * (1 + max_degradation)
    )

    passed = candidate_metric <= allowed_metric

    return {
        "passed": passed,
        "candidate_metric": candidate_metric,
        "incumbent_metric": incumbent_metric,
        "allowed_metric": allowed_metric,
        "max_degradation": max_degradation,
    }