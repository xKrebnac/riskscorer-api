# RiskScorer API

ML-Portfolio-Projekt: Kreditrisikovorhersage mit scikit-learn und FastAPI.

---

## Projektstruktur

```
riskscorer-api/
├── src/
│   ├── preprocessor.py   # Datenlade-, Analyse- und Vorverarbeitungslogik
│   ├── trainer.py        # Modelltraining, Evaluation, Speichern/Laden
│   ├── api.py            # FastAPI REST-Endpunkte
│   ├── cli.py            # Kommandozeilenschnittstelle
│   └── main.py           # Uvicorn-Starter
├── tests/
│   ├── test_preprocessor.py
│   ├── test_trainer.py
│   └── test_api.py
├── data/
│   ├── raw/              # Rohdaten (credit_risk_dataset.csv)
│   └── processed/        # Verarbeitete Daten
├── models/               # Gespeicherte Modelle (.joblib)
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Phase 1 – Datenanalyse

```python
from src.preprocessor import CreditRiskPreprocessor

p = CreditRiskPreprocessor()
df = p.load_data("data/raw/credit_risk_dataset.csv")
p.explore(df)                # Shape, Typen, fehlende Werte, Klassenverteilung
X, y = p.preprocess(df)     # Fehlende Werte auffüllen, One-Hot-Encoding, X/y trennen
X_train, X_test, y_train, y_test = p.split(X, y)
```

---

## Phase 2 – Modelltraining

```python
from src.trainer import RiskModelTrainer

trainer = RiskModelTrainer()
trainer.train(X_train, y_train, model_type="gradient_boosting")
metrics = trainer.evaluate(X_test, y_test)
# {'accuracy': 0.9312, 'precision': 0.8741, 'recall': 0.8203, 'f1': 0.8463, 'roc_auc': 0.9587}

trainer.save("models/risk_model.joblib")
```

Unterstützte Modelltypen: `gradient_boosting`, `random_forest`, `logistic`

---

## Phase 3 – REST API

### Starten

```bash
python src/main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### Endpunkte

| Methode | Pfad       | Beschreibung                                      |
|---------|------------|---------------------------------------------------|
| GET     | `/health`  | API-Status und ob ein Modell geladen ist          |
| POST    | `/predict` | Vorhersage für einen einzelnen Kreditantrag       |
| POST    | `/train`   | Modell trainieren, evaluieren und speichern       |

### Beispiel: Vorhersage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 28,
    "person_income": 45000,
    "person_emp_length": 3,
    "loan_amnt": 8000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.18,
    "cb_person_cred_hist_length": 3,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "cb_person_default_on_file": "N"
  }'
```

```json
{
  "prediction": 0,
  "probability": 0.1423,
  "risk_label": "Geringes Risiko"
}
```

### Beispiel: Training via API

```bash
curl -X POST "http://localhost:8000/train?model_type=random_forest"
```

---

## Phase 4 – CLI

```bash
# Modell trainieren
python -m src.cli train \
  --data data/raw/credit_risk_dataset.csv \
  --model gradient_boosting \
  --output models/risk_model.joblib

# Gespeichertes Modell evaluieren
python -m src.cli evaluate \
  --data data/raw/credit_risk_dataset.csv \
  --model-path models/risk_model.joblib
```

---

## Tests

```bash
# Alle Tests
python -m pytest tests/ -v

# Einzelne Testdatei
python -m pytest tests/test_preprocessor.py -v
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_api.py -v
```

---

## Datensatz

Erwartet wird `data/raw/credit_risk_dataset.csv` mit diesen Spalten:

| Spalte                       | Typ       | Beschreibung                        |
|------------------------------|-----------|-------------------------------------|
| `person_age`                 | float     | Alter der antragstellenden Person   |
| `person_income`              | float     | Jahreseinkommen                     |
| `person_emp_length`          | float     | Beschäftigungsdauer in Jahren       |
| `loan_amnt`                  | float     | Kreditbetrag                        |
| `loan_int_rate`              | float     | Zinssatz in Prozent                 |
| `loan_percent_income`        | float     | Kreditbetrag / Einkommen            |
| `cb_person_cred_hist_length` | float     | Länge der Kredithistorie in Jahren  |
| `loan_intent`                | str       | Kreditzweck (EDUCATION, MEDICAL...) |
| `loan_grade`                 | str       | Kreditrating (A–G)                  |
| `cb_person_default_on_file`  | str       | Frühere Zahlungsausfälle (Y/N)      |
| `loan_status`                | int       | Zielspalte: 0 = kein Ausfall, 1 = Ausfall |
