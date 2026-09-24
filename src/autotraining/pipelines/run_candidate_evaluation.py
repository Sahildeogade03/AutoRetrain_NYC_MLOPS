from __future__ import annotations

from autotraining.pipelines.retrain import (
    train_and_evaluate_candidate,
    evaluate_candidate_quality,
)
from autotraining.pipelines.train import (
    load_config,
    load_zone_panel,
)


INCUMBENT_MAE = 11.378989


def main() -> None:

    print(
        "AutoRetrain-NYC | Candidate Evaluation"
    )

    print("=" * 60)

    config = load_config()

    model_config = config.get(
        "model",
        {},
    )

    device = model_config.get(
        "device",
        "cpu",
    )

    horizon = int(
        model_config.get(
            "horizon",
            336,
        )
    )

    panel = load_zone_panel()

    result = train_and_evaluate_candidate(
        panel=panel,
        horizon=horizon,
        device=device,
    )

    quality = evaluate_candidate_quality(
        candidate_mae=result["mae"],
        incumbent_mae=INCUMBENT_MAE,
    )

    print("\nCandidate results")
    print("-" * 60)

    print(
        f"MAE:  {result['mae']:.6f}"
    )

    print(
        f"RMSE: {result['rmse']:.6f}"
    )

    print(
        f"Zones: {result['zones']}"
    )

    print(
        f"Holdout: "
        f"{result['holdout_start']} "
        f"→ "
        f"{result['holdout_end']}"
    )

    print("\nQuality gate")
    print("-" * 60)

    print(
        f"Candidate MAE: "
        f"{quality['candidate_metric']:.6f}"
    )

    print(
        f"Incumbent MAE: "
        f"{quality['incumbent_metric']:.6f}"
    )

    print(
        f"Allowed MAE: "
        f"{quality['allowed_metric']:.6f}"
    )

    print(
        f"Result: "
        f"{'PASS' if quality['passed'] else 'FAIL'}"
    )

    print(
        f"Action: "
        f"{quality['action'].upper()}"
    )


if __name__ == "__main__":
    main()