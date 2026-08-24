from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml.pipelines import (
    SEED,
    build_candidates,
    build_pipeline,
    build_preprocessor,
)
from src.ml.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EXPECTED_CANDIDATES = {
    "logistic_regression",
    "logistic_regression_balanced",
    "random_forest",
    "mlp",
}


def test_build_preprocessor_covers_every_feature():
    preprocessor = build_preprocessor()
    columns = [column for _, _, columns in preprocessor.transformers for column in columns]

    assert isinstance(preprocessor, ColumnTransformer)
    assert columns == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_build_preprocessor_ignores_unknown_categories():
    encoder = dict(
        (name, transformer) for name, transformer, _ in build_preprocessor().transformers
    )["cat"]

    assert encoder.handle_unknown == "ignore"


def test_build_pipeline_has_preprocessor_then_classifier():
    pipeline = build_pipeline(LogisticRegression())

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]


def test_build_candidates_returns_the_expected_models():
    assert set(build_candidates()) == EXPECTED_CANDIDATES


def test_build_candidates_does_not_share_the_preprocessor():
    candidates = build_candidates()
    preprocessors = {
        id(pipeline.named_steps["preprocessor"]) for pipeline in candidates.values()
    }

    assert len(preprocessors) == len(candidates)


def test_build_candidates_uses_the_given_seed():
    candidates = build_candidates(seed=7)

    for pipeline in candidates.values():
        assert pipeline.named_steps["classifier"].random_state == 7


def test_build_candidates_defaults_to_the_project_seed():
    classifier = build_candidates()["random_forest"].named_steps["classifier"]

    assert classifier.random_state == SEED


def test_balanced_candidate_weights_the_minority_class():
    classifier = build_candidates()["logistic_regression_balanced"].named_steps["classifier"]

    assert classifier.class_weight == "balanced"
