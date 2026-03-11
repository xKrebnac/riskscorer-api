"""Tests für die FastAPI-Endpunkte der RiskScorer API."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app

VALID_LOAN_REQUEST = {
    "person_age": 30.0,
    "person_income": 60000.0,
    "person_emp_length": 5.0,
    "loan_amnt": 10000.0,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 4.0,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "cb_person_default_on_file": "N",
}


@pytest.fixture
def client():
    """TestClient ohne geladenes Modell (Standardzustand)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_model():
    """TestClient mit gemocktem, trainiertem Modell."""
    with TestClient(app) as c:
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([1])
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
        c.app.state.trainer.pipeline_ = mock_pipeline
        c.app.state.feature_names = None  # Kein Reindexing beim Mock
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_endpoint_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_correct_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_not_loaded(self, client):
        """model_loaded ist False wenn kein Modell geladen wurde."""
        data = client.get("/health").json()
        assert data["model_loaded"] is False

    def test_health_model_loaded(self, client_with_model):
        """model_loaded ist True wenn ein Modell im State liegt."""
        data = client_with_model.get("/health").json()
        assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_without_model_returns_503(self, client):
        response = client.post("/predict", json=VALID_LOAN_REQUEST)
        assert response.status_code == 503

    def test_predict_returns_200_with_model(self, client_with_model):
        response = client_with_model.post("/predict", json=VALID_LOAN_REQUEST)
        assert response.status_code == 200

    def test_predict_returns_correct_keys(self, client_with_model):
        data = client_with_model.post("/predict", json=VALID_LOAN_REQUEST).json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_label" in data

    def test_predict_high_risk_label(self, client_with_model):
        """prediction=1 soll risk_label 'Hohes Risiko' ergeben."""
        data = client_with_model.post("/predict", json=VALID_LOAN_REQUEST).json()
        assert data["prediction"] == 1
        assert data["risk_label"] == "Hohes Risiko"

    def test_predict_low_risk_label(self, client_with_model):
        """prediction=0 soll risk_label 'Geringes Risiko' ergeben."""
        client_with_model.app.state.trainer.pipeline_.predict.return_value = np.array([0])
        client_with_model.app.state.trainer.pipeline_.predict_proba.return_value = np.array(
            [[0.85, 0.15]]
        )
        data = client_with_model.post("/predict", json=VALID_LOAN_REQUEST).json()
        assert data["prediction"] == 0
        assert data["risk_label"] == "Geringes Risiko"

    def test_predict_invalid_input_returns_422(self, client_with_model):
        """Fehlende Pflichtfelder sollen einen 422-Fehler auslösen."""
        response = client_with_model.post("/predict", json={"person_age": 30.0})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /train
# ---------------------------------------------------------------------------


class TestTrainEndpoint:
    def test_train_without_data_returns_404(self, client):
        """Fehlende CSV-Datei soll einen 404-Fehler auslösen."""
        response = client.post("/train")
        assert response.status_code == 404

    def test_train_unknown_model_type_returns_422(self, client, tmp_path, monkeypatch):
        """Unbekannter model_type soll einen 422-Fehler auslösen."""
        # Datendatei simulieren damit der 404 nicht zuerst greift
        import src.api as api_module

        monkeypatch.setattr(api_module, "DATA_PATH", tmp_path / "credit_risk_dataset.csv")
        (tmp_path / "credit_risk_dataset.csv").write_text(
            "person_age,loan_status\n25,0\n35,1\n"
        )
        response = client.post("/train?model_type=svm")
        assert response.status_code == 422
