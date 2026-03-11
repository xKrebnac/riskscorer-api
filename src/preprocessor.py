import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CreditRiskPreprocessor:
    """Vorverarbeitungs-Pipeline für Kreditrisikodaten."""

    def load_data(self, filepath: str) -> pd.DataFrame:
        """CSV-Datei einlesen und als DataFrame zurückgeben."""
        df = pd.read_csv(filepath)
        return df

    def explore(self, df: pd.DataFrame) -> None:
        """Grundlegende EDA ausgeben: Shape, Datentypen, fehlende Werte, Klassenverteilung."""
        logger.info("=== Shape ===\n%s", df.shape)

        logger.info("=== Spaltentypen ===\n%s", df.dtypes.to_string())

        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_report = pd.DataFrame({"fehlend": missing, "pct": missing_pct})
        missing_report = missing_report[missing_report["fehlend"] > 0]
        logger.info("=== Fehlende Werte pro Spalte ===\n%s", missing_report.to_string())

        if "loan_status" in df.columns:
            counts = df["loan_status"].value_counts()
            pcts = df["loan_status"].value_counts(normalize=True).mul(100).round(2)
            dist = pd.DataFrame({"Anzahl": counts, "pct": pcts})
            dist.index = dist.index.map({0: "0 (kein Ausfall)", 1: "1 (Ausfall)"})
            logger.info("=== Klassenverteilung (loan_status) ===\n%s", dist.to_string())

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fehlende Werte auffüllen, kategorische Spalten kodieren, X und y trennen."""
        df = df.copy()

        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        df = pd.get_dummies(df, columns=list(categorical_cols), drop_first=False)

        y = df["loan_status"]
        X = df.drop(columns=["loan_status"])
        return X, y

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Daten stratifiziert in Trainings- und Testmenge aufteilen."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(
            "Aufteilung: %d Trainings- und %d Testsamples (test_size=%.2f)",
            len(X_train),
            len(X_test),
            test_size,
        )
        return X_train, X_test, y_train, y_test
