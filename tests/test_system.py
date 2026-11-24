"""
Automated Test Suite for Predictive Maintenance 6G Smart Manufacturing Platform
Author: Syed Abdul Rehman (Register No: 230071601233)
Organization: Unified Mentor Pvt. Ltd. / B.S. Abdur Rahman Crescent Institute

Run with:
    python -m unittest tests/test_system.py
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "processed_dataset.csv")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "model_artifacts.joblib")
BRANDING_DIR = os.path.join(BASE_DIR, "assets", "branding")


class TestEnvironmentAndDependencies(unittest.TestCase):
    """Verifies that all required packages and dependencies are installed and accessible."""

    def test_imports(self):
        import pandas as pd
        import numpy as np
        import sklearn
        import streamlit as st
        import plotly
        import joblib
        self.assertTrue(hasattr(pd, "__version__"))
        self.assertTrue(hasattr(np, "__version__"))
        self.assertTrue(hasattr(sklearn, "__version__"))
        self.assertTrue(hasattr(st, "__version__"))
        self.assertTrue(hasattr(plotly, "__version__"))
        self.assertTrue(hasattr(joblib, "__version__"))


class TestBrandingAssets(unittest.TestCase):
    """Verifies existence and non-emptiness of all vector SVG branding assets."""

    def test_svg_assets_exist(self):
        required_assets = [
            "project_logo.svg",
            "project_logo_dark.svg",
            "project_symbol.svg",
            "favicon.svg"
        ]
        for asset in required_assets:
            path = os.path.join(BRANDING_DIR, asset)
            self.assertTrue(os.path.exists(path), f"Missing brand asset: {asset}")
            self.assertGreater(os.path.getsize(path), 100, f"Asset is empty or too small: {asset}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("<svg", content, f"Asset does not contain valid SVG tag: {asset}")


class TestDatasetSchemaAndIntegrity(unittest.TestCase):
    """Validates raw and processed dataset dimensions, columns, and data hygiene."""

    def setUp(self):
        self.assertTrue(os.path.exists(DATASET_PATH), "dataset.csv not found!")
        self.df_raw = pd.read_csv(DATASET_PATH)

    def test_raw_dimensions_and_counts(self):
        self.assertEqual(self.df_raw.shape[0], 10800, "dataset.csv must have exactly 10,800 rows")
        self.assertEqual(self.df_raw.shape[1], 14, "dataset.csv must have exactly 14 columns")

    def test_no_null_values_or_duplicates(self):
        self.assertEqual(self.df_raw.isnull().sum().sum(), 0, "dataset.csv contains null values")
        self.assertEqual(self.df_raw.duplicated().sum(), 0, "dataset.csv contains duplicate records")

    def test_machines_and_date_bounds(self):
        machines = sorted(self.df_raw["Machine_ID"].unique().tolist())
        expected_machines = [f"M_{i:03d}" for i in range(1, 11)]
        self.assertEqual(machines, expected_machines, "Machine IDs must be M_001 through M_010")

        # Each machine must have exactly 1,080 readings (36 records/day x 30 days)
        counts = self.df_raw["Machine_ID"].value_counts().to_dict()
        for m, count in counts.items():
            self.assertEqual(count, 1080, f"Machine {m} has {count} records instead of 1,080")

        self.assertEqual(self.df_raw["Date"].min(), "2026-06-01")
        self.assertEqual(self.df_raw["Date"].max(), "2026-06-30")

    def test_processed_dataset_schema(self):
        self.assertTrue(os.path.exists(PROCESSED_PATH), "processed_dataset.csv not found!")
        df_proc = pd.read_csv(PROCESSED_PATH)
        self.assertEqual(df_proc.shape[0], 10800, "processed_dataset.csv must have 10,800 rows")
        self.assertEqual(df_proc.shape[1], 23, "processed_dataset.csv must have 23 columns")

        # Anomaly score bounds
        self.assertTrue("Anomaly_Score" in df_proc.columns)
        self.assertGreaterEqual(df_proc["Anomaly_Score"].min(), 0.0)
        self.assertLessEqual(df_proc["Anomaly_Score"].max(), 100.0)

        # Risk level distribution
        self.assertTrue("Maintenance_Risk_Level" in df_proc.columns)
        categories = set(df_proc["Maintenance_Risk_Level"].unique())
        self.assertEqual(categories, {"Low", "Medium", "High"})


class TestModelArtifactsAndInference(unittest.TestCase):
    """Validates the serialized Isolation Forest model, scaler, and inference capability."""

    def setUp(self):
        self.assertTrue(os.path.exists(ARTIFACTS_PATH), "model_artifacts.joblib not found!")
        self.artifacts = joblib.load(ARTIFACTS_PATH)

    def test_artifact_structure(self):
        self.assertIn("model", self.artifacts)
        self.assertIn("scaler", self.artifacts)
        self.assertIn("feature_cols", self.artifacts)

        model = self.artifacts["model"]
        self.assertEqual(model.n_estimators, 150)
        self.assertEqual(model.contamination, 0.08)
        self.assertEqual(model.random_state, 42)

        feature_cols = self.artifacts["feature_cols"]
        self.assertEqual(len(feature_cols), 11)

    def test_sample_inference(self):
        model = self.artifacts["model"]
        scaler = self.artifacts["scaler"]
        feature_cols = self.artifacts["feature_cols"]

        # Synthetic sample vector
        dummy_sample = pd.DataFrame([{col: 0.0 for col in feature_cols}])
        raw_score = -model.decision_function(dummy_sample)
        scaled_score = scaler.transform(raw_score.reshape(-1, 1))[0][0]

        self.assertIsInstance(float(scaled_score), float)
        self.assertGreaterEqual(scaled_score, 0.0)
        self.assertLessEqual(scaled_score, 100.0)


class TestAppModule(unittest.TestCase):
    """Verifies that app.py syntax compiles cleanly and can be inspected without execution crashes."""

    def test_app_compiles(self):
        import py_compile
        app_path = os.path.join(BASE_DIR, "app.py")
        compiled = py_compile.compile(app_path, doraise=True)
        self.assertTrue(os.path.exists(compiled))


if __name__ == "__main__":
    unittest.main()
