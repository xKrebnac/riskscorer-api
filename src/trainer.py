import logging

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("gradient_boosting", "random_forest", "logistic")


class RiskModelTrainer:
    """Trainiert, evaluiert und speichert Kreditrisikomodelle."""

    def __init__(self) -> None:
        self.pipeline_: Pipeline | None = None

    def train(self, X_train, y_train, model_type: str = "gradient_boosting") -> None:
        """Sklearn-Pipeline mit StandardScaler und gewähltem Modell trainieren.

        Args:
            X_train: Trainingsfeatures.
            y_train: Trainingslabels.
            model_type: Modelltyp – 'gradient_boosting', 'random_forest' oder 'logistic'.

        Raises:
            ValueError: Bei unbekanntem model_type.
        """
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unbekannter Modelltyp '{model_type}'. "
                f"Unterstützt: {SUPPORTED_MODELS}"
            )

        logger.info("Starte Training mit Modelltyp '%s'.", model_type)

        if model_type == "gradient_boosting":
            # GradientBoosting unterstützt kein class_weight; sample_weight wäre nötig,
            # aber für die Pipeline-Schnittstelle verwenden wir hier den Standard.
            model = GradientBoostingClassifier(random_state=42)
        elif model_type == "random_forest":
            model = RandomForestClassifier(class_weight="balanced", random_state=42)
        else:  # logistic
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )

        self.pipeline_ = Pipeline(
            steps=[("scaler", StandardScaler()), ("model", model)]
        )
        self.pipeline_.fit(X_train, y_train)
        logger.info("Training abgeschlossen.")

    def evaluate(self, X_test, y_test) -> dict:
        """Modell auf Testdaten evaluieren und Metriken als Dict zurückgeben.

        Returns:
            Dict mit accuracy, precision, recall, f1, roc_auc (je 4 Dezimalstellen).

        Raises:
            RuntimeError: Wenn das Modell noch nicht trainiert wurde.
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "Modell wurde noch nicht trainiert. Bitte zuerst train() aufrufen."
            )

        y_pred = self.pipeline_.predict(X_test)
        y_proba = self.pipeline_.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }

        logger.info(
            "Evaluierungsergebnisse – Accuracy: %.4f | Precision: %.4f | "
            "Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        return metrics

    def save(self, filepath: str) -> None:
        """Trainiertes Modell mit joblib auf Disk speichern.

        Raises:
            RuntimeError: Wenn das Modell noch nicht trainiert wurde.
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "Kein trainiertes Modell vorhanden. Bitte zuerst train() aufrufen."
            )
        joblib.dump(self.pipeline_, filepath)
        logger.info("Modell gespeichert unter '%s'.", filepath)

    def load(self, filepath: str) -> None:
        """Gespeichertes Modell von Disk laden und in self.pipeline_ setzen."""
        self.pipeline_ = joblib.load(filepath)
        logger.info("Modell geladen von '%s'.", filepath)
