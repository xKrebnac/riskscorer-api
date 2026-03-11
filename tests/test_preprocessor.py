import logging

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import CreditRiskPreprocessor


@pytest.fixture
def preprocessor():
    return CreditRiskPreprocessor()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 35, np.nan, 45],
            "income": [50000.0, np.nan, 72000.0, 60000.0],
            "home_ownership": ["RENT", "OWN", np.nan, "MORTGAGE"],
            "loan_status": [0, 1, 0, 1],
        }
    )


class TestLoadData:
    def test_returns_dataframe(self, preprocessor, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        df = preprocessor.load_data(str(csv_file))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_file_not_found(self, preprocessor):
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("nonexistent.csv")


class TestExplore:
    def test_runs_without_error(self, preprocessor, sample_df, caplog):
        with caplog.at_level(logging.INFO):
            preprocessor.explore(sample_df)
        assert "Shape" in caplog.text
        assert "Spaltentypen" in caplog.text
        assert "Fehlende Werte" in caplog.text
        assert "loan_status" in caplog.text

    def test_no_loan_status_column(self, preprocessor, caplog):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with caplog.at_level(logging.INFO):
            preprocessor.explore(df)
        assert "Klassenverteilung" not in caplog.text


class TestPreprocess:
    def test_returns_tuple(self, preprocessor, sample_df):
        result = preprocessor.preprocess(sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_missing_values_after_preprocess(self, preprocessor, sample_df):
        X, y = preprocessor.preprocess(sample_df)
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_numeric_filled_with_median(self, preprocessor):
        df = pd.DataFrame({"value": [1.0, 3.0, np.nan], "loan_status": [0, 1, 0]})
        X, y = preprocessor.preprocess(df)
        assert X["value"].iloc[2] == pytest.approx(2.0)

    def test_categorical_filled_with_mode(self, preprocessor):
        df = pd.DataFrame({"cat": ["A", "A", "B", np.nan], "loan_status": [0, 1, 0, 1]})
        X, y = preprocessor.preprocess(df)
        assert "cat_A" in X.columns
        assert X["cat_A"].iloc[3] == 1

    def test_categorical_encoded_with_dummies(self, preprocessor, sample_df):
        X, y = preprocessor.preprocess(sample_df)
        assert "home_ownership" not in X.columns
        assert any(col.startswith("home_ownership_") for col in X.columns)

    def test_loan_status_separated(self, preprocessor, sample_df):
        X, y = preprocessor.preprocess(sample_df)
        assert "loan_status" not in X.columns
        assert list(y) == [0, 1, 0, 1]

    def test_original_df_not_mutated(self, preprocessor, sample_df):
        original_missing = sample_df.isnull().sum().sum()
        preprocessor.preprocess(sample_df)
        assert sample_df.isnull().sum().sum() == original_missing


class TestSplit:
    @pytest.fixture
    def X_y(self, preprocessor, sample_df):
        # Datensatz vergrößern damit stratify funktioniert (mind. 2 Samples pro Klasse)
        df = pd.concat([sample_df] * 10, ignore_index=True)
        return preprocessor.preprocess(df)

    def test_returns_four_parts(self, preprocessor, X_y):
        X, y = X_y
        result = preprocessor.split(X, y)
        assert len(result) == 4

    def test_split_sizes(self, preprocessor, X_y):
        X, y = X_y
        X_train, X_test, y_train, y_test = preprocessor.split(X, y, test_size=0.2)
        total = len(X)
        assert len(X_test) == pytest.approx(total * 0.2, abs=1)
        assert len(X_train) + len(X_test) == total

    def test_stratification_preserved(self, preprocessor, X_y):
        X, y = X_y
        _, _, y_train, y_test = preprocessor.split(X, y)
        assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)

    def test_reproducibility(self, preprocessor, X_y):
        X, y = X_y
        result1 = preprocessor.split(X, y, random_state=42)
        result2 = preprocessor.split(X, y, random_state=42)
        pd.testing.assert_frame_equal(result1[0], result2[0])
