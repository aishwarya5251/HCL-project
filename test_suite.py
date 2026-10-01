"""
Comprehensive automated test suite for the modified 100% Local Conversational Data Assistant.
"""

import os
import unittest
import pandas as pd
from engine.data_loader import load_csv, get_dataset_summary
from engine.local_nlp_engine import LocalNLPEngine
from engine.query_executor import QueryExecutor


class TestLocalDataAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = os.path.join(os.path.dirname(__file__), "samples", "ecommerce_sales.csv")
        cls.df = load_csv(cls.sample_path)
        cls.nlp = LocalNLPEngine(cls.df, table_name="dataset")
        cls.executor = QueryExecutor(cls.df, table_name="dataset")

    def test_data_ingestion_and_health(self):
        self.assertFalse(self.df.empty)
        summary = get_dataset_summary(self.df)
        self.assertGreater(summary["rows"], 0)
        self.assertIn("Sales", summary["numeric_cols"])
        self.assertIn("Region", summary["categorical_cols"])

    def test_ranking_nl_query(self):
        res = self.nlp.query("What are the top 5 sub-categories by sales?")
        self.assertTrue(bool(res.get("sql")))
        self.assertIsNotNone(res.get("fig"))
        self.assertEqual(len(res.get("result_df")), 5)

    def test_time_series_nl_query(self):
        res = self.nlp.query("Show monthly sales trend over time")
        self.assertTrue(bool(res.get("sql")))
        self.assertIsNotNone(res.get("fig"))
        self.assertGreater(len(res.get("result_df")), 0)

    def test_correlation_nl_query(self):
        res = self.nlp.query("Is there correlation between discount and profit?")
        self.assertIsNotNone(res.get("fig"))
        self.assertIn("Pearson Correlation Coefficient", res.get("explanation", ""))

    def test_outlier_detection_nl_query(self):
        res = self.nlp.query("Detect outliers in profit")
        self.assertTrue(bool(res.get("sql")))
        self.assertIsNotNone(res.get("fig"))

    def test_composition_nl_query(self):
        res = self.nlp.query("Share of sales by region")
        self.assertTrue(bool(res.get("sql")))
        self.assertIsNotNone(res.get("fig"))

    def test_duckdb_direct_execution(self):
        sql = 'SELECT "Customer_Segment", COUNT(*) AS Total FROM dataset GROUP BY "Customer_Segment";'
        res = self.executor.execute_sql(sql)
        self.assertTrue(res["success"])
        self.assertGreater(len(res["data"]), 0)


if __name__ == "__main__":
    unittest.main()
