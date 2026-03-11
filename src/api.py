"""FastAPI-Anwendung für die RiskScorer API."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.preprocessor import CreditRiskPreprocessor
from src.trainer import RiskModelTrainer

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/risk_model.joblib")
FEATURE_NAMES_PATH = Path("models/feature_names.json")
DATA_PATH = Path("data/raw/credit_risk_dataset.csv")

CATEGORICAL_COLS = ["loan_intent", "loan_grade", "cb_person_default_on_file"]


# ---------------------------------------------------------------------------
# Pydantic-Modelle
# ---------------------------------------------------------------------------


class LoanRequest(BaseModel):
    """Eingabedaten für einen einzelnen Kreditantrag."""

    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    loan_intent: str
    loan_grade: str
    cb_person_default_on_file: str


class PredictionResponse(BaseModel):
    """Vorhersageergebnis für einen Kreditantrag."""

    prediction: int
    probability: float
    risk_label: str


class HealthResponse(BaseModel):
    """Statusantwort der API."""

    status: str
    model_loaded: bool


class TrainResponse(BaseModel):
    """Ergebnis eines Trainingsaufrufs."""

    message: str
    model_type: str
    metrics: dict


# ---------------------------------------------------------------------------
# Lebenszyklus
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modell beim Start laden, falls vorhanden."""
    trainer = RiskModelTrainer()
    app.state.trainer = trainer
    app.state.feature_names = None

    if MODEL_PATH.exists():
        try:
            trainer.load(str(MODEL_PATH))
            logger.info("Modell geladen von '%s'.", MODEL_PATH)
            if FEATURE_NAMES_PATH.exists():
                with open(FEATURE_NAMES_PATH, encoding="utf-8") as f:
                    app.state.feature_names = json.load(f)
        except Exception as exc:
            logger.warning("Modell konnte nicht geladen werden: %s", exc)

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RiskScorer API",
    description="Kreditrisikovorhersage mit Machine Learning.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpunkte
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Gibt den Betriebsstatus der API und den Ladezustand des Modells zurück."""
    return HealthResponse(
        status="ok",
        model_loaded=app.state.trainer.pipeline_ is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: LoanRequest) -> PredictionResponse:
    """Vorhersage für einen einzelnen Kreditantrag erstellen.

    Raises:
        HTTPException 503: Wenn kein Modell geladen ist.
    """
    trainer: RiskModelTrainer = app.state.trainer
    if trainer.pipeline_ is None:
        raise HTTPException(
            status_code=503,
            detail="Kein Modell geladen. Bitte zuerst POST /train aufrufen.",
        )

    # Anfrage in DataFrame umwandeln und kategorische Spalten kodieren
    row = request.model_dump()
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Auf Trainings-Features ausrichten (fehlende Dummy-Spalten mit 0 auffüllen)
    if app.state.feature_names:
        df = df.reindex(columns=app.state.feature_names, fill_value=0)

    prediction = int(trainer.pipeline_.predict(df)[0])
    probability = round(float(trainer.pipeline_.predict_proba(df)[0][1]), 4)
    risk_label = "Hohes Risiko" if prediction == 1 else "Geringes Risiko"

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        risk_label=risk_label,
    )


@app.post("/train", response_model=TrainResponse)
def train(
    model_type: str = Query(
        default="gradient_boosting",
        description="Modelltyp: 'gradient_boosting', 'random_forest' oder 'logistic'.",
    ),
) -> TrainResponse:
    """Neues Modell auf dem Kreditrisiko-Datensatz trainieren und speichern.

    Raises:
        HTTPException 404: Wenn die Datendatei nicht gefunden wird.
        HTTPException 422: Bei ungültigem model_type.
        HTTPException 500: Bei unerwarteten Fehlern während des Trainings.
    """
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Datendatei nicht gefunden: {DATA_PATH}",
        )

    preprocessor = CreditRiskPreprocessor()
    trainer: RiskModelTrainer = app.state.trainer

    try:
        df = preprocessor.load_data(str(DATA_PATH))
        X, y = preprocessor.preprocess(df)
        X_train, X_test, y_train, y_test = preprocessor.split(X, y)
        trainer.train(X_train, y_train, model_type=model_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trainingsfehler: {exc}") from exc

    metrics = trainer.evaluate(X_test, y_test)

    # Modell und Feature-Namen speichern
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(MODEL_PATH))
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f)
    app.state.feature_names = list(X_train.columns)

    return TrainResponse(
        message="Modell erfolgreich trainiert und gespeichert.",
        model_type=model_type,
        metrics=metrics,
    )
