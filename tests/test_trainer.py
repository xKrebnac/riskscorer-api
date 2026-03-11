import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

from src.trainer import RiskModelTrainer

EXPECTED_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


@pytest.fixture
def trainer():
    return RiskModelTrainer()


@pytest.fixture
def classification_data():
    """Synthetischer Binärklassifikationsdatensatz für alle Trainer-Tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    split = 160
    return X[:split], X[split:], y[:split], y[split:]


class TestTrain:
    def test_train_gradient_boosting(self, trainer, classification_data):
        X_train, _, y_train, _ = classification_data
        trainer.train(X_train, y_train, model_type="gradient_boosting")
        assert isinstance(trainer.pipeline_, Pipeline)

    def test_train_random_forest(self, trainer, classification_data):
        X_train, _, y_train, _ = classification_data
        trainer.train(X_train, y_train, model_type="random_forest")
        assert isinstance(trainer.pipeline_, Pipeline)

    def test_train_logistic(self, trainer, classification_data):
        X_train, _, y_train, _ = classification_data
        trainer.train(X_train, y_train, model_type="logistic")
        assert isinstance(trainer.pipeline_, Pipeline)

    def test_train_unknown_model_raises(self, trainer, classification_data):
        X_train, _, y_train, _ = classification_data
        with pytest.raises(ValueError, match="Unbekannter Modelltyp"):
            trainer.train(X_train, y_train, model_type="svm")

    def test_pipeline_initialized_to_none(self):
        assert RiskModelTrainer().pipeline_ is None


class TestEvaluate:
    def test_evaluate_returns_correct_keys(self, trainer, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        assert set(metrics.keys()) == EXPECTED_METRIC_KEYS

    def test_evaluate_values_in_range(self, trainer, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} außerhalb [0, 1]: {value}"

    def test_evaluate_values_rounded_to_4_decimals(self, trainer, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        for key, value in metrics.items():
            assert value == round(value, 4), f"{key} nicht auf 4 Stellen gerundet"

    def test_evaluate_before_train_raises(self, trainer, classification_data):
        _, X_test, _, y_test = classification_data
        with pytest.raises(RuntimeError):
            trainer.evaluate(X_test, y_test)


class TestSaveLoad:
    def test_save_and_load(self, trainer, classification_data, tmp_path):
        X_train, X_test, y_train, y_test = classification_data
        trainer.train(X_train, y_train)
        filepath = str(tmp_path / "model.joblib")

        trainer.save(filepath)

        new_trainer = RiskModelTrainer()
        new_trainer.load(filepath)
        assert new_trainer.pipeline_ is not None

        metrics = new_trainer.evaluate(X_test, y_test)
        assert set(metrics.keys()) == EXPECTED_METRIC_KEYS

    def test_save_before_train_raises(self, trainer, tmp_path):
        with pytest.raises(RuntimeError):
            trainer.save(str(tmp_path / "model.joblib"))
