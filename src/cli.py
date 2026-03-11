"""Kommandozeilenschnittstelle für Training und Evaluation des RiskScorer-Modells.

Verwendung:
    python -m src.cli train    --data data/raw/credit_risk_dataset.csv
    python -m src.cli evaluate --data data/raw/credit_risk_dataset.csv \\
                                --model-path models/risk_model.joblib
"""

import argparse
import logging
import sys

from src.preprocessor import CreditRiskPreprocessor
from src.trainer import RiskModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktion: Metriken ausgeben
# ---------------------------------------------------------------------------


def _print_metrics(metrics: dict) -> None:
    """Evaluierungsmetriken formatiert auf der Konsole ausgeben."""
    print("\n" + "=" * 40)
    print("  Evaluierungsergebnisse")
    print("=" * 40)
    labels = {
        "accuracy":  "Accuracy ",
        "precision": "Precision",
        "recall":    "Recall   ",
        "f1":        "F1-Score ",
        "roc_auc":   "ROC-AUC  ",
    }
    for key, label in labels.items():
        bar_len = int(metrics[key] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label}  {bar}  {metrics[key]:.4f}")
    print("=" * 40 + "\n")


# ---------------------------------------------------------------------------
# Befehl: train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Vollständigen Trainingsablauf ausführen und Modell speichern."""
    preprocessor = CreditRiskPreprocessor()
    trainer = RiskModelTrainer()

    logger.info("Lade Datensatz von '%s'.", args.data)
    try:
        df = preprocessor.load_data(args.data)
    except FileNotFoundError:
        logger.error("Datei nicht gefunden: %s", args.data)
        sys.exit(1)

    logger.info("Vorverarbeitung gestartet.")
    X, y = preprocessor.preprocess(df)
    logger.info("Features: %d Spalten, %d Zeilen.", X.shape[1], X.shape[0])

    X_train, X_test, y_train, y_test = preprocessor.split(X, y)

    logger.info("Starte Training (Modelltyp: '%s').", args.model)
    try:
        trainer.train(X_train, y_train, model_type=args.model)
    except ValueError as exc:
        logger.error("Ungültiger Modelltyp: %s", exc)
        sys.exit(1)

    metrics = trainer.evaluate(X_test, y_test)
    _print_metrics(metrics)

    trainer.save(args.output)
    logger.info("Modell gespeichert unter '%s'.", args.output)


# ---------------------------------------------------------------------------
# Befehl: evaluate
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Gespeichertes Modell auf einem Datensatz evaluieren."""
    preprocessor = CreditRiskPreprocessor()
    trainer = RiskModelTrainer()

    logger.info("Lade Datensatz von '%s'.", args.data)
    try:
        df = preprocessor.load_data(args.data)
    except FileNotFoundError:
        logger.error("Datei nicht gefunden: %s", args.data)
        sys.exit(1)

    logger.info("Vorverarbeitung gestartet.")
    X, y = preprocessor.preprocess(df)

    logger.info("Lade Modell von '%s'.", args.model_path)
    try:
        trainer.load(args.model_path)
    except FileNotFoundError:
        logger.error("Modelldatei nicht gefunden: %s", args.model_path)
        sys.exit(1)

    metrics = trainer.evaluate(X, y)
    _print_metrics(metrics)


# ---------------------------------------------------------------------------
# Argument-Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Argparse-Parser mit Unterbefehlen aufbauen."""
    parser = argparse.ArgumentParser(
        prog="src.cli",
        description="RiskScorer – Kreditrisikomodell trainieren und evaluieren.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- train ---------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train", help="Modell trainieren und speichern."
    )
    train_parser.add_argument(
        "--data",
        default="data/raw/credit_risk_dataset.csv",
        help="Pfad zur CSV-Datendatei (Standard: data/raw/credit_risk_dataset.csv).",
    )
    train_parser.add_argument(
        "--model",
        default="gradient_boosting",
        choices=["gradient_boosting", "random_forest", "logistic"],
        help="Zu trainierender Modelltyp (Standard: gradient_boosting).",
    )
    train_parser.add_argument(
        "--output",
        default="models/risk_model.joblib",
        help="Speicherort für das trainierte Modell (Standard: models/risk_model.joblib).",
    )
    train_parser.set_defaults(func=cmd_train)

    # -- evaluate ------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "evaluate", help="Gespeichertes Modell auf einem Datensatz evaluieren."
    )
    eval_parser.add_argument(
        "--data",
        default="data/raw/credit_risk_dataset.csv",
        help="Pfad zur CSV-Datendatei (Standard: data/raw/credit_risk_dataset.csv).",
    )
    eval_parser.add_argument(
        "--model-path",
        default="models/risk_model.joblib",
        help="Pfad zur gespeicherten Modelldatei (Standard: models/risk_model.joblib).",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    """Einstiegspunkt für den CLI-Aufruf."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
