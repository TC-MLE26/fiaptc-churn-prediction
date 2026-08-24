"""Training pipeline: loads the dataset, compares candidates and saves the champion."""

from __future__ import annotations
import json
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from src.ml.data import load_raw
from src.ml.evaluation import compare_models, cross_validate, select_champion
from src.ml.pipelines import SEED, build_candidates
from src.ml.preprocessing import clean_data, split_features_target

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CHAMPION_PATH = MODELS_DIR / "champion_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
TEST_SIZE = 0.2


def train(seed: int = SEED, test_size: float = TEST_SIZE):
    features, target = split_features_target(clean_data(load_raw()))

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=seed
    )

    candidates = build_candidates(seed)
    validation = {
        name: cross_validate(model, features_train, target_train, seed=seed)
        for name, model in candidates.items()
    }

    fitted = {
        name: model.fit(features_train, target_train)
        for name, model in candidates.items()
    }

    comparison = compare_models(fitted, features_test, target_test, validation)
    champion_name = select_champion(comparison)

    return fitted[champion_name], champion_name, comparison


def save(model, champion_name: str, comparison) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CHAMPION_PATH)

    import sklearn

    METRICS_PATH.write_text(
        json.dumps(
            {
                "champion": champion_name,
                "seed": SEED,
                "sklearn_version": sklearn.__version__,
                "comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model, champion_name, comparison = train()
    save(model, champion_name, comparison)

    logger.info("%s", comparison.to_string(index=False))
    logger.info("champion: %s", champion_name)
    logger.info("saved to: %s", CHAMPION_PATH)


if __name__ == "__main__":
    main()
