"""
Query and Visualization Executor for Conversational Data Analysis.
Supports DuckDB SQL execution and Plotly chart generation.
"""

import time
import re
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go


class QueryExecutor:
    def __init__(self, df: pd.DataFrame, table_name: str = "dataset"):
        self.df = df
        self.table_name = table_name
        self.con = duckdb.connect(database=":memory:")
        # Register the dataframe in DuckDB
        self.con.register(self.table_name, self.df)

    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Execute a DuckDB SQL query and return the result dataframe and metadata."""
        # Clean query: strip code blocks if present
        clean_query = self._sanitize_code(sql_query, "sql")
        start_time = time.perf_counter()

        try:
            # DuckDB query
            result_df = self.con.execute(clean_query).df()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "success": True,
                "data": result_df,
                "rows": len(result_df),
                "cols": len(result_df.columns),
                "elapsed_ms": elapsed_ms,
                "error": None,
                "query": clean_query,
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "success": False,
                "data": None,
                "rows": 0,
                "cols": 0,
                "elapsed_ms": elapsed_ms,
                "error": str(e),
                "query": clean_query,
            }

    def execute_chart_code(
        self, chart_code: str, sql_result_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Execute Plotly visualization code in a safe execution context.
        Provides `df`, `result_df`, `pd`, `np`, `px`, and `go` in scope.
        The code is expected to assign the Plotly figure to a variable named `fig`.
        """
        clean_code = self._sanitize_code(chart_code, "python")
        if not clean_code.strip():
            return {"success": False, "fig": None, "error": "No chart code provided"}

        local_scope = {
            "df": self.df,
            "result_df": sql_result_df if sql_result_df is not None else self.df,
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "fig": None,
        }

        try:
            # Execute python code
            exec(clean_code, {}, local_scope)
            fig = local_scope.get("fig", None)
            if fig is None:
                return {
                    "success": False,
                    "fig": None,
                    "error": "The visualization code did not assign a Plotly figure to the variable `fig`.",
                }
            # Standardize layout style
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40),
                hoverlabel=dict(bgcolor="white", font_size=12),
            )
            return {"success": True, "fig": fig, "error": None, "code": clean_code}
        except Exception as e:
            return {"success": False, "fig": None, "error": str(e), "code": clean_code}

    def fallback_analysis(self, query_text: str) -> Dict[str, Any]:
        """
        Rule-based heuristic analysis for basic queries when LLM is unavailable or offline.
        """
        q = query_text.lower().strip()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in self.df.columns if c not in numeric_cols]

        # 1. Summary / Describe
        if any(w in q for w in ["summary", "describe", "overview", "statistics"]):
            desc = self.df.describe(include="all").reset_index()
            return {
                "thought": "Generated statistical summary across all columns.",
                "explanation": f"Here is the statistical summary of **{self.table_name}** covering {len(self.df)} rows and {len(self.df.columns)} columns.",
                "sql": f"SELECT * FROM {self.table_name} LIMIT 10",
                "result_df": desc,
                "chart_code": None,
                "fig": None,
            }

        # 2. Missing values
        if any(w in q for w in ["null", "missing", "nan", "empty"]):
            null_counts = self.df.isna().sum().reset_index()
            null_counts.columns = ["Column", "Missing_Count"]
            null_counts["Missing_Percentage"] = round((null_counts["Missing_Count"] / len(self.df)) * 100, 2)
            fig = px.bar(
                null_counts,
                x="Column",
                y="Missing_Count",
                title="Missing Values per Column",
                text="Missing_Count",
                color="Missing_Count",
                color_continuous_scale="Reds",
            )
            return {
                "thought": "Calculated null/missing count per column.",
                "explanation": "Here is the missing values breakdown across all columns in the dataset.",
                "sql": f"SELECT column_name, null_count FROM ...",
                "result_df": null_counts,
                "chart_code": "# Generated Plotly bar chart for missing values",
                "fig": fig,
            }

        # 3. Correlation
        if "correlat" in q and len(numeric_cols) >= 2:
            corr = self.df[numeric_cols].corr().round(3)
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap (Numeric Columns)",
                color_continuous_scale="Blues",
            )
            return {
                "thought": "Computed Pearson correlation matrix for numeric columns.",
                "explanation": "Here is the correlation matrix for the numeric attributes in the dataset.",
                "sql": None,
                "result_df": corr.reset_index(),
                "chart_code": "# Generated correlation heatmap",
                "fig": fig,
            }

        # 4. Top / distribution of first categorical column
        if cat_cols:
            primary_cat = cat_cols[0]
            for col in cat_cols:
                if col.lower() in q:
                    primary_cat = col
                    break
            
            top_df = self.df[primary_cat].value_counts().reset_index().head(10)
            top_df.columns = [primary_cat, "Count"]
            fig = px.bar(
                top_df,
                x=primary_cat,
                y="Count",
                title=f"Distribution of {primary_cat} (Top 10)",
                text="Count",
                color="Count",
                color_continuous_scale="Viridis",
            )
            return {
                "thought": f"Calculated top occurrences for column '{primary_cat}'.",
                "explanation": f"Here is the breakdown of the top values for **{primary_cat}**.",
                "sql": f'SELECT "{primary_cat}", COUNT(*) AS Count FROM {self.table_name} GROUP BY "{primary_cat}" ORDER BY Count DESC LIMIT 10;',
                "result_df": top_df,
                "chart_code": f'fig = px.bar(result_df, x="{primary_cat}", y="Count", title="Distribution of {primary_cat}")',
                "fig": fig,
            }

        # Default sample preview
        preview = self.df.head(10)
        return {
            "thought": "Extracted top 10 preview rows.",
            "explanation": f"Displaying preview of the first 10 rows. To get tailored SQL analysis and custom charts, ensure a Gemini API key is configured.",
            "sql": f"SELECT * FROM {self.table_name} LIMIT 10;",
            "result_df": preview,
            "chart_code": None,
            "fig": None,
        }

    @staticmethod
    def _sanitize_code(code_str: str, language: str) -> str:
        """Strip markdown code fence blocks if present."""
        if not code_str:
            return ""
        s = code_str.strip()
        # Remove ```sql ... ``` or ```python ... ``` or ``` ... ```
        pattern = rf"^```(?:{language}|py|sql)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, s, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return s.strip()
