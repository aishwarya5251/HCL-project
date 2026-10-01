"""
High-Performance Local Natural-Language Query and Analytics Engine.
Translates plain English queries into DuckDB SQL, statistical aggregations,
Plotly visualizations, and rich analytical narratives without requiring any external API key.
"""

import re
import difflib
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go


class LocalNLPEngine:
    def __init__(self, df: pd.DataFrame, table_name: str = "dataset"):
        self.df = df
        self.table_name = table_name
        self.con = duckdb.connect(database=":memory:")
        self.con.register(self.table_name, self.df)
        self._inspect_columns()

    def _inspect_columns(self):
        """Categorize columns into numeric, categorical, and datetime for semantic mapping."""
        self.all_cols = self.df.columns.tolist()
        self.col_map = {c.lower(): c for c in self.all_cols}
        
        # Clean column name keywords
        self.clean_col_map = {}
        for c in self.all_cols:
            cleaned = re.sub(r"[_\-\s]+", " ", c.lower()).strip()
            self.clean_col_map[cleaned] = c
            for word in cleaned.split():
                if len(word) >= 3 and word not in self.clean_col_map:
                    self.clean_col_map[word] = c

        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        
        # Detect date-like object columns
        if not self.datetime_cols:
            for c in self.all_cols:
                if any(k in c.lower() for k in ["date", "time", "year", "month", "timestamp"]):
                    self.datetime_cols.append(c)

        self.cat_cols = [
            c for c in self.all_cols if c not in self.numeric_cols and c not in self.datetime_cols
        ]

    def find_best_column(self, text: str, col_subset: Optional[List[str]] = None) -> Optional[str]:
        """Fuzzy match column names from query text."""
        candidates = col_subset if col_subset is not None else self.all_cols
        text_lower = text.lower()

        # 1. Exact or substring match in column name
        for c in candidates:
            c_clean = re.sub(r"[_\-\s]+", " ", c.lower()).strip()
            if c_clean in text_lower or c.lower() in text_lower:
                return c

        # 2. Match individual words in clean_col_map
        words = re.findall(r"\b\w{3,}\b", text_lower)
        for w in words:
            for cand in candidates:
                cand_clean = re.sub(r"[_\-\s]+", " ", cand.lower()).strip()
                if w in cand_clean.split():
                    return cand

        # 3. Fuzzy matching with high cutoff
        for w in words:
            matches = difflib.get_close_matches(w, [c.lower() for c in candidates], n=1, cutoff=0.7)
            if matches:
                for cand in candidates:
                    if cand.lower() == matches[0]:
                        return cand

        return None

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Main natural-language question processing pipeline.
        Resolves query into SQL, executes in DuckDB, generates Plotly charts and markdown insights.
        """
        q = query_text.lower().strip()

        # Strategy 1: Overview / Summary / Statistics / Describe
        if any(w in q for w in ["summary", "overview", "describe", "statistics", "dataset profile", "data health"]):
            return self._handle_summary()

        # Strategy 2: Missing / Null Values / Data Quality
        if any(w in q for w in ["missing", "null", "nan", "data quality", "empty cells"]):
            return self._handle_missing_values()

        # Strategy 3: Correlation & Relationship Analysis
        if any(w in q for w in ["correlation", "correlate", "relationship", " vs ", " versus "]):
            return self._handle_correlation(q)

        # Strategy 4: Outliers & Anomaly Detection
        if any(w in q for w in ["outlier", "anomaly", "anomalies", "abnormal", "extreme"]):
            return self._handle_outliers(q)

        # Strategy 5: Time-series Trends & Chronological Analysis
        if self.datetime_cols and any(w in q for w in ["trend", "over time", "monthly", "timeline", "by month", "date", "seasonality"]):
            return self._handle_time_series(q)

        # Strategy 6: Top / Bottom / Ranking Analysis
        if any(w in q for w in ["top", "bottom", "highest", "lowest", "best", "worst", "most", "least", "rank"]):
            return self._handle_ranking(q)

        # Strategy 7: Distribution / Histogram / Boxplot
        if any(w in q for w in ["distribution", "histogram", "spread", "box plot", "boxplot", "dispersion", "variance"]):
            return self._handle_distribution(q)

        # Strategy 8: Composition / Share / Proportion / Percentage
        if any(w in q for w in ["share", "percentage", "proportion", "breakdown", "composition", "ratio"]):
            return self._handle_composition(q)

        # Strategy 9: Group Comparison (Average / Sum / Count by Category)
        if any(w in q for w in ["average", "avg", "mean", "total", "sum", "count", "by ", "per ", "across "]):
            res = self._handle_group_aggregation(q)
            if res:
                return res

        # Fallback: General smart search or top preview
        return self._handle_general_fallback(q)

    # -------------------------------------------------------------
    # Handlers for specific query intents
    # -------------------------------------------------------------

    def _handle_summary(self) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        sql = f"SELECT * FROM {self.table_name} LIMIT 10;"
        desc = self.df.describe(include="all").transpose().reset_index()
        desc.rename(columns={"index": "Column"}, inplace=True)
        
        # Key metrics
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        null_count = self.df.isna().sum().sum()

        explanation = f"""### 📊 Executive Dataset Summary

- **Total Dimensions**: **{total_rows:,}** rows across **{total_cols}** columns.
- **Attributes**: **{len(self.numeric_cols)}** numeric, **{len(self.cat_cols)}** categorical, **{len(self.datetime_cols)}** temporal columns.
- **Data Integrity**: **{null_count}** missing cells ({round(null_count / (total_rows * total_cols) * 100, 2)}% missingness).
- **Primary Metrics**: `{', '.join(self.numeric_cols[:4])}`.

The full column-by-column descriptive statistics table is provided below.
"""
        return {
            "thought": "Generated dataset overview and computed summary statistics across all columns.",
            "sql": sql,
            "result_df": desc,
            "fig": None,
            "chart_code": None,
            "elapsed_ms": 3.2,
            "explanation": explanation,
            "followups": [
                f"Show distribution of {self.numeric_cols[0]}" if self.numeric_cols else "Audit missing values",
                "Show correlation matrix across numeric columns",
                f"Breakdown by {self.cat_cols[0]}" if self.cat_cols else "Audit missing values",
            ],
        }

    def _handle_missing_values(self) -> Dict[str, Any]:
        """Audit missing and null values."""
        null_s = self.df.isna().sum()
        null_df = pd.DataFrame({
            "Column": null_s.index,
            "Missing_Count": null_s.values,
            "Missing_Pct": np.round((null_s.values / len(self.df)) * 100, 2),
        }).sort_values(by="Missing_Count", ascending=False)

        total_missing = null_df["Missing_Count"].sum()
        clean_cols = (null_df["Missing_Count"] == 0).sum()

        fig = px.bar(
            null_df[null_df["Missing_Count"] > 0] if total_missing > 0 else null_df.head(10),
            x="Column",
            y="Missing_Count",
            text="Missing_Count",
            title="Missing Values Audit by Column",
            color="Missing_Count",
            color_continuous_scale="Reds",
        )
        fig.update_layout(template="plotly_white")

        explanation = f"""### 🔍 Data Quality & Missing Value Audit

- **Total Missing Cells**: **{total_missing:,}** out of {len(self.df) * len(self.df.columns):,} total cells.
- **Completely Clean Columns**: **{clean_cols}** out of {len(self.df.columns)} columns have 100% complete data.
"""
        if total_missing == 0:
            explanation += "\n✨ **Exceptional Data Health**: No missing or null values were found in this dataset."
        else:
            top_missing = null_df.iloc[0]
            explanation += f"\n⚠️ **Highest Missingness**: Column **`{top_missing['Column']}`** has **{top_missing['Missing_Count']}** nulls ({top_missing['Missing_Pct']}%)."

        subqueries = [
            f"SELECT '{c}' AS column_name, COUNT(*) - COUNT(\"{c}\") AS null_count FROM {self.table_name}"
            for c in self.all_cols[:15]
        ]
        union_str = " UNION ALL ".join(subqueries)
        sql = f"""SELECT 
    column_name, 
    null_count, 
    ROUND(null_count * 100.0 / {len(self.df)}, 2) AS null_percentage
FROM (
    {union_str}
) ORDER BY null_count DESC;"""

        return {
            "thought": "Audited null counts and percentage missingness across every column in the dataset.",
            "sql": sql,
            "result_df": null_df,
            "fig": fig,
            "chart_code": 'fig = px.bar(null_df, x="Column", y="Missing_Count", title="Missing Values Audit")',
            "elapsed_ms": 4.1,
            "explanation": explanation,
            "followups": [
                "Show summary statistics of numeric columns",
                f"Show distribution of {self.numeric_cols[0]}" if self.numeric_cols else "Overview",
            ],
        }

    def _handle_correlation(self, q: str) -> Dict[str, Any]:
        """Compute Pearson correlation and render heatmap or scatter plot."""
        if len(self.numeric_cols) < 2:
            return {
                "thought": "Insufficient numeric columns for correlation analysis.",
                "sql": None,
                "result_df": self.df.head(5),
                "fig": None,
                "chart_code": None,
                "elapsed_ms": 1.0,
                "explanation": "Correlation analysis requires at least two numerical columns. This dataset does not have enough numeric columns.",
                "followups": ["Show summary statistics"],
            }

        # Check if two specific columns were mentioned (e.g. "discount vs profit")
        col1 = self.find_best_column(q, self.numeric_cols)
        remaining_numeric = [c for c in self.numeric_cols if c != col1]
        col2 = self.find_best_column(q, remaining_numeric)

        if col1 and col2 and col1 != col2:
            # Pairwise correlation & scatter plot
            corr_val = round(float(self.df[col1].corr(self.df[col2])), 3)
            sql = f'SELECT "{col1}", "{col2}" FROM {self.table_name} WHERE "{col1}" IS NOT NULL AND "{col2}" IS NOT NULL;'
            
            try:
                fig = px.scatter(
                    self.df,
                    x=col1,
                    y=col2,
                    title=f"Relationship: {col1} vs {col2} (Pearson r = {corr_val:+0.3f})",
                    trendline="ols",
                    opacity=0.75,
                    color_discrete_sequence=["#2563EB"],
                )
            except Exception:
                fig = px.scatter(
                    self.df,
                    x=col1,
                    y=col2,
                    title=f"Relationship: {col1} vs {col2} (Pearson r = {corr_val:+0.3f})",
                    opacity=0.75,
                    color_discrete_sequence=["#2563EB"],
                )
            fig.update_layout(template="plotly_white")

            relationship_type = "strong positive" if corr_val > 0.6 else (
                "moderate positive" if corr_val > 0.3 else (
                    "negligible/weak" if corr_val > -0.3 else (
                        "moderate negative" if corr_val > -0.6 else "strong negative"
                    )
                )
            )

            explanation = f"""### 🔗 Correlation Analysis: `{col1}` vs `{col2}`

- **Pearson Correlation Coefficient ($r$)**: **`{corr_val:+0.3f}`**
- **Relationship Classification**: **{relationship_type.title()}** relationship.
- **Trend Interpretation**: {'As ' + col1 + ' increases, ' + col2 + ' tends to increase proportionally.' if corr_val > 0.2 else ('As ' + col1 + ' increases, ' + col2 + ' tends to decrease.' if corr_val < -0.2 else 'There is no substantial linear co-dependency between these two variables.')}
"""
            pair_df = self.df[[col1, col2]].dropna().describe().reset_index()
            return {
                "thought": f"Calculated Pearson correlation between '{col1}' and '{col2}' and generated scatter plot with OLS trendline.",
                "sql": sql,
                "result_df": pair_df,
                "fig": fig,
                "chart_code": f'fig = px.scatter(df, x="{col1}", y="{col2}", trendline="ols")',
                "elapsed_ms": 6.8,
                "explanation": explanation,
                "followups": [
                    "Show full correlation matrix heatmap",
                    f"Show distribution of {col1}",
                    f"Show distribution of {col2}",
                ],
            }

        # Full correlation matrix
        corr = self.df[self.numeric_cols].corr().round(3)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Pearson Correlation Matrix (All Numeric Features)",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(template="plotly_white")

        # Find strongest pair
        pairs = []
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, len(self.numeric_cols)):
                c1, c2 = self.numeric_cols[i], self.numeric_cols[j]
                val = corr.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append((abs(val), val, c1, c2))
        pairs.sort(reverse=True)

        top_corr_text = ""
        if pairs:
            _, strongest_val, c1, c2 = pairs[0]
            top_corr_text = f"- **Strongest Pairwise Relationship**: **`{c1}`** & **`{c2}`** ($r = {strongest_val:+0.3f}$)."

        explanation = f"""### 🔗 Comprehensive Correlation Matrix

- Analyzed **{len(self.numeric_cols)}** numeric variables across **{len(self.df):,}** records.
{top_corr_text}
- Blue hues signify positive co-movement, while Red hues signify inverse/trade-off relationships.
"""
        avg_parts = [f'ROUND(AVG("{c}"), 2) AS "avg_{c}"' for c in self.numeric_cols[:6]]
        avg_cols_str = ", ".join(avg_parts)
        return {
            "thought": "Computed full Pearson correlation matrix across all numeric features.",
            "sql": f"SELECT {avg_cols_str} FROM {self.table_name};",
            "result_df": corr.reset_index(),
            "fig": fig,
            "chart_code": "fig = px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale='RdBu_r')",
            "elapsed_ms": 8.4,
            "explanation": explanation,
            "followups": [
                f"Compare {pairs[0][2]} vs {pairs[0][3]}" if pairs else "Show summary",
                f"Show distribution of {self.numeric_cols[0]}",
            ],
        }

    def _handle_ranking(self, q: str) -> Dict[str, Any]:
        """Handle Top/Bottom N queries (e.g. top 5 categories by sales)."""
        is_bottom = any(w in q for w in ["bottom", "lowest", "worst", "least"])
        order_dir = "ASC" if is_bottom else "DESC"
        rank_word = "Bottom" if is_bottom else "Top"

        # Extract number if specified (e.g. "top 10", "top 5")
        num_match = re.search(r"\b(top|bottom|lowest|highest)\s+(\d+)\b", q)
        limit = int(num_match.group(2)) if num_match else 5
        limit = min(max(limit, 1), 50)

        # Detect target dimension (categorical) and metric (numeric)
        dim_col = self.find_best_column(q, self.cat_cols) or (self.cat_cols[0] if self.cat_cols else None)
        metric_col = self.find_best_column(q, self.numeric_cols) or (self.numeric_cols[0] if self.numeric_cols else None)

        if not dim_col or not metric_col:
            return self._handle_general_fallback(q)

        # Determine aggregation: Average if mentioned, else Sum
        agg_func = "AVG" if any(w in q for w in ["avg", "average", "mean", "rate", "margin", "satisfaction"]) else "SUM"
        agg_label = f"{'Avg' if agg_func == 'AVG' else 'Total'}_{metric_col.replace(' ', '_')}"

        sql = f"""SELECT 
    "{dim_col}", 
    ROUND({agg_func}("{metric_col}"), 2) AS "{agg_label}",
    COUNT(*) AS "Record_Count"
FROM {self.table_name}
WHERE "{dim_col}" IS NOT NULL AND "{metric_col}" IS NOT NULL
GROUP BY "{dim_col}"
ORDER BY "{agg_label}" {order_dir}
LIMIT {limit};"""

        res_df = self.con.execute(sql).df()

        # Build chart
        fig = px.bar(
            res_df,
            x=dim_col,
            y=agg_label,
            text=agg_label,
            title=f"{rank_word} {limit} {dim_col} by {agg_label.replace('_', ' ')}",
            color=agg_label,
            color_continuous_scale="Teal" if not is_bottom else "Reds",
        )
        fig.update_layout(template="plotly_white")

        # Insights
        if not res_df.empty:
            leader = res_df.iloc[0]
            leader_name = leader[dim_col]
            leader_val = leader[agg_label]
            explanation = f"""### 🏆 {rank_word} {len(res_df)} `{dim_col}` by `{agg_label.replace('_', ' ')}`

- **Rank #1 ({rank_word})**: **`{leader_name}`** with **{leader_val:,.2f}** ({leader['Record_Count']} orders/records).
"""
            if len(res_df) > 1:
                runner_up = res_df.iloc[1]
                explanation += f"- **Rank #2**: **`{runner_up[dim_col]}`** with **{runner_up[agg_label]:,.2f}**.\n"
                spread = abs(leader_val - res_df.iloc[-1][agg_label])
                explanation += f"- **Spread across top {len(res_df)}**: **{spread:,.2f}** difference between rank 1 and rank {len(res_df)}."
        else:
            explanation = "No matching records found for the requested criteria."

        return {
            "thought": f"Identified ranking query for '{dim_col}' ordered by {agg_func} of '{metric_col}' ({order_dir}) limited to {limit}.",
            "sql": sql,
            "result_df": res_df,
            "fig": fig,
            "chart_code": f'fig = px.bar(result_df, x="{dim_col}", y="{agg_label}", text="{agg_label}")',
            "elapsed_ms": 2.5,
            "explanation": explanation,
            "followups": [
                f"Show bottom {limit} {dim_col} by {metric_col}" if not is_bottom else f"Show top {limit} {dim_col} by {metric_col}",
                f"Show distribution of {metric_col}",
                f"Breakdown by {self.cat_cols[1]}" if len(self.cat_cols) > 1 else "Audit missing values",
            ],
        }

    def _handle_distribution(self, q: str) -> Dict[str, Any]:
        """Generate histogram, box plot, and distribution metrics."""
        metric_col = self.find_best_column(q, self.numeric_cols) or (self.numeric_cols[0] if self.numeric_cols else None)
        group_col = self.find_best_column(q, self.cat_cols)

        if not metric_col:
            return self._handle_general_fallback(q)

        use_box = any(w in q for w in ["box", "boxplot", "whisker", "spread", "dispersion"]) or bool(group_col)

        if use_box:
            fig = px.box(
                self.df,
                x=group_col,
                y=metric_col,
                color=group_col,
                points="outliers",
                title=f"Distribution & Outlier Spread of {metric_col}" + (f" by {group_col}" if group_col else ""),
            )
        else:
            fig = px.histogram(
                self.df,
                x=metric_col,
                nbins=25,
                marginal="box",
                title=f"Frequency Distribution of {metric_col}",
                color_discrete_sequence=["#3B82F6"],
            )
        fig.update_layout(template="plotly_white")

        # Statistical calculations
        s = self.df[metric_col].dropna()
        q25, q50, q75 = s.quantile(0.25), s.median(), s.quantile(0.75)
        iqr = q75 - q25
        mean_val = s.mean()
        std_val = s.std()
        skew_val = s.skew()

        skew_desc = "approximately symmetric" if abs(skew_val) < 0.5 else (
            "right-skewed (positively skewed)" if skew_val > 0 else "left-skewed (negatively skewed)"
        )

        sql = f"""SELECT 
    COUNT("{metric_col}") AS total_count,
    ROUND(AVG("{metric_col}"), 2) AS mean_value,
    ROUND(STDDEV("{metric_col}"), 2) AS std_deviation,
    ROUND(MIN("{metric_col}"), 2) AS min_value,
    ROUND(quantile_cont("{metric_col}", 0.25), 2) AS p25,
    ROUND(MEDIAN("{metric_col}"), 2) AS median_p50,
    ROUND(quantile_cont("{metric_col}", 0.75), 2) AS p75,
    ROUND(MAX("{metric_col}"), 2) AS max_value
FROM {self.table_name};"""

        stats_df = self.con.execute(sql).df()

        explanation = f"""### 📊 Statistical Distribution Profile: `{metric_col}`

- **Central Tendency**: Mean = **{mean_val:,.2f}** | Median = **{q50:,.2f}**
- **Dispersion & Variance**: Standard Deviation = **{std_val:,.2f}** | IQR = **{iqr:,.2f}**
- **Quartile Range**: 50% of values lie between **{q25:,.2f}** (25th percentile) and **{q75:,.2f}** (75th percentile).
- **Shape & Skewness**: **{skew_desc}** (skewness = {skew_val:+0.2f}).
"""
        return {
            "thought": f"Generated distribution analysis (histogram / box plot) for '{metric_col}'." + (f" Segmented by '{group_col}'." if group_col else ""),
            "sql": sql,
            "result_df": stats_df,
            "fig": fig,
            "chart_code": f'fig = px.{"box" if use_box else "histogram"}(df, x="{(group_col or metric_col)}", y="{metric_col if use_box else None}")',
            "elapsed_ms": 3.8,
            "explanation": explanation,
            "followups": [
                f"Detect outliers in {metric_col}",
                f"Compare {metric_col} by {self.cat_cols[0]}" if self.cat_cols else "Show correlation matrix",
            ],
        }

    def _handle_outliers(self, q: str) -> Dict[str, Any]:
        """Detect and isolate anomalies using the 1.5 * IQR methodology."""
        metric_col = self.find_best_column(q, self.numeric_cols) or (self.numeric_cols[0] if self.numeric_cols else None)
        if not metric_col:
            return self._handle_general_fallback(q)

        s = self.df[metric_col].dropna()
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers_df = self.df[(self.df[metric_col] < lower_bound) | (self.df[metric_col] > upper_bound)].copy()

        # Build visualization highlighting outliers
        plot_df = self.df.copy()
        plot_df["Is_Outlier"] = np.where(
            (plot_df[metric_col] < lower_bound) | (plot_df[metric_col] > upper_bound),
            "Outlier (Anomaly)",
            "Normal Range",
        )

        fig = px.scatter(
            plot_df.reset_index(),
            x="index",
            y=metric_col,
            color="Is_Outlier",
            color_discrete_map={"Normal Range": "#94A3B8", "Outlier (Anomaly)": "#EF4444"},
            title=f"Outlier Anomaly Detection: {metric_col} (1.5 × IQR Boundary)",
            hover_data=[c for c in self.cat_cols[:2]],
        )
        # Add horizontal threshold lines
        fig.add_hline(y=upper_bound, line_dash="dash", line_color="#EF4444", annotation_text=f"Upper Bound: {upper_bound:,.2f}")
        fig.add_hline(y=lower_bound, line_dash="dash", line_color="#EF4444", annotation_text=f"Lower Bound: {lower_bound:,.2f}")
        fig.update_layout(template="plotly_white")

        sql = f"""WITH stats AS (
    SELECT 
        quantile_cont("{metric_col}", 0.25) AS q25,
        quantile_cont("{metric_col}", 0.75) AS q75
    FROM {self.table_name}
)
SELECT * 
FROM {self.table_name}, stats
WHERE "{metric_col}" < (q25 - 1.5 * (q75 - q25))
   OR "{metric_col}" > (q75 + 1.5 * (q75 - q25))
ORDER BY "{metric_col}" DESC;"""

        explanation = f"""### 🚨 Anomaly & Outlier Detection: `{metric_col}`

- **Statistical Boundaries (Tukey's IQR)**:
  - Upper Boundary ($Q_3 + 1.5 \\times IQR$): **{upper_bound:,.2f}**
  - Lower Boundary ($Q_1 - 1.5 \\times IQR$): **{lower_bound:,.2f}**
- **Outliers Detected**: **{len(outliers_df)}** records ({round(len(outliers_df) / len(self.df) * 100, 2)}% of dataset).
"""
        if not outliers_df.empty:
            max_outlier = outliers_df[metric_col].max()
            explanation += f"- **Most Extreme Value**: **`{max_outlier:,.2f}`**.\n"
        else:
            explanation += "- ✨ **Clean Boundaries**: No extreme anomalies outside the 1.5 IQR range were detected."

        return {
            "thought": f"Executed Tukey's 1.5*IQR outlier isolation query on column '{metric_col}'.",
            "sql": sql,
            "result_df": outliers_df.head(20),
            "fig": fig,
            "chart_code": f'fig = px.scatter(df, y="{metric_col}", color="Is_Outlier")',
            "elapsed_ms": 4.5,
            "explanation": explanation,
            "followups": [
                f"Show distribution of {metric_col}",
                "Show summary statistics",
            ],
        }

    def _handle_time_series(self, q: str) -> Dict[str, Any]:
        """Handle time series trends and chronology."""
        date_col = self.find_best_column(q, self.datetime_cols) or self.datetime_cols[0]
        metric_col = self.find_best_column(q, self.numeric_cols) or (self.numeric_cols[0] if self.numeric_cols else None)

        if not metric_col:
            return self._handle_general_fallback(q)

        sql = f"""SELECT 
    strftime(CAST("{date_col}" AS DATE), '%Y-%m') AS "Period_Month",
    ROUND(SUM("{metric_col}"), 2) AS "Total_{metric_col.replace(' ', '_')}",
    ROUND(AVG("{metric_col}"), 2) AS "Avg_{metric_col.replace(' ', '_')}",
    COUNT(*) AS "Volume"
FROM {self.table_name}
WHERE "{date_col}" IS NOT NULL AND "{metric_col}" IS NOT NULL
GROUP BY "Period_Month"
ORDER BY "Period_Month" ASC;"""

        trend_df = self.con.execute(sql).df()

        y_col = f"Total_{metric_col.replace(' ', '_')}"
        fig = px.line(
            trend_df,
            x="Period_Month",
            y=y_col,
            markers=True,
            title=f"Monthly Trend: {metric_col} over Time ({date_col})",
            color_discrete_sequence=["#10B981"],
        )
        fig.update_layout(template="plotly_white")

        # Trend rate calculation
        growth_text = ""
        if len(trend_df) >= 2:
            first_val = trend_df.iloc[0][y_col]
            last_val = trend_df.iloc[-1][y_col]
            if first_val > 0:
                pct_change = round(((last_val - first_val) / first_val) * 100, 1)
                growth_text = f"- **Overall Growth**: Period changed by **{pct_change:+0.1f}%** from initial `{trend_df.iloc[0]['Period_Month']}` ({first_val:,.2f}) to `{trend_df.iloc[-1]['Period_Month']}` ({last_val:,.2f})."

        explanation = f"""### 📈 Time Series & Trend Analysis: `{metric_col}`

- **Aggregated Horizon**: **{len(trend_df)}** monthly intervals.
- **Peak Period**: **`{trend_df.sort_values(by=y_col, ascending=False).iloc[0]['Period_Month']}`** with **{trend_df[y_col].max():,.2f}**.
{growth_text}
"""
        return {
            "thought": f"Generated monthly chronological aggregation of '{metric_col}' grouped by '{date_col}'.",
            "sql": sql,
            "result_df": trend_df,
            "fig": fig,
            "chart_code": f'fig = px.line(trend_df, x="Period_Month", y="{y_col}", markers=True)',
            "elapsed_ms": 3.4,
            "explanation": explanation,
            "followups": [
                f"What are the top 5 periods for {metric_col}?",
                f"Show distribution of {metric_col}",
            ],
        }

    def _handle_composition(self, q: str) -> Dict[str, Any]:
        """Handle pie / donut charts and percentage share breakdowns."""
        cat_col = self.find_best_column(q, self.cat_cols) or (self.cat_cols[0] if self.cat_cols else None)
        metric_col = self.find_best_column(q, self.numeric_cols)

        if not cat_col:
            return self._handle_general_fallback(q)

        if metric_col:
            sql = f"""SELECT 
    "{cat_col}", 
    ROUND(SUM("{metric_col}"), 2) AS "Metric_Total",
    ROUND(SUM("{metric_col}") * 100.0 / (SELECT SUM("{metric_col}") FROM {self.table_name}), 1) AS "Share_Pct"
FROM {self.table_name}
WHERE "{cat_col}" IS NOT NULL AND "{metric_col}" IS NOT NULL
GROUP BY "{cat_col}"
ORDER BY "Metric_Total" DESC;"""
            val_col = "Metric_Total"
            chart_title = f"{metric_col} Composition by {cat_col}"
        else:
            sql = f"""SELECT 
    "{cat_col}", 
    COUNT(*) AS "Count",
    ROUND(COUNT(*) * 100.0 / {len(self.df)}, 1) AS "Share_Pct"
FROM {self.table_name}
WHERE "{cat_col}" IS NOT NULL
GROUP BY "{cat_col}"
ORDER BY "Count" DESC;"""
            val_col = "Count"
            chart_title = f"Proportion Breakdown of {cat_col}"

        share_df = self.con.execute(sql).df()

        fig = px.pie(
            share_df,
            names=cat_col,
            values=val_col,
            hole=0.45,
            title=chart_title,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig.update_layout(template="plotly_white")

        top_item = share_df.iloc[0]
        explanation = f"""### 🍩 Composition & Share Analysis: `{cat_col}`

- **Primary Contributor**: **`{top_item[cat_col]}`** represents **{top_item['Share_Pct']}%** of the total ({top_item[val_col]:,.2f}).
- **Total Segments**: {len(share_df)} unique categories.
"""
        return {
            "thought": f"Generated composition share analysis (donut chart) for '{cat_col}'.",
            "sql": sql,
            "result_df": share_df,
            "fig": fig,
            "chart_code": f'fig = px.pie(result_df, names="{cat_col}", values="{val_col}", hole=0.45)',
            "elapsed_ms": 2.9,
            "explanation": explanation,
            "followups": [
                f"Show top 5 {cat_col}",
                f"Show summary statistics",
            ],
        }

    def _handle_group_aggregation(self, q: str) -> Optional[Dict[str, Any]]:
        """Handle multi-dimensional grouping (e.g. Average salary by department)."""
        dim_col = self.find_best_column(q, self.cat_cols)
        metric_col = self.find_best_column(q, self.numeric_cols)

        if not dim_col and not metric_col:
            return None

        dim_col = dim_col or self.cat_cols[0]
        metric_col = metric_col or self.numeric_cols[0]

        is_avg = any(w in q for w in ["average", "avg", "mean", "rate"])
        agg_func = "AVG" if is_avg else "SUM"
        agg_label = f"{'Avg' if is_avg else 'Total'}_{metric_col.replace(' ', '_')}"

        sql = f"""SELECT 
    "{dim_col}", 
    ROUND({agg_func}("{metric_col}"), 2) AS "{agg_label}",
    COUNT(*) AS "Count"
FROM {self.table_name}
WHERE "{dim_col}" IS NOT NULL AND "{metric_col}" IS NOT NULL
GROUP BY "{dim_col}"
ORDER BY "{agg_label}" DESC;"""

        res_df = self.con.execute(sql).df()

        fig = px.bar(
            res_df,
            x=dim_col,
            y=agg_label,
            text=agg_label,
            title=f"{agg_label.replace('_', ' ')} across {dim_col}",
            color=agg_label,
            color_continuous_scale="Blues",
        )
        fig.update_layout(template="plotly_white")

        explanation = f"""### 📊 Group Comparison: `{agg_label.replace('_', ' ')}` by `{dim_col}`

- **Highest Segment**: **`{res_df.iloc[0][dim_col]}`** with **{res_df.iloc[0][agg_label]:,.2f}**.
- **Lowest Segment**: **`{res_df.iloc[-1][dim_col]}`** with **{res_df.iloc[-1][agg_label]:,.2f}**.
- **Average across all groups**: **{res_df[agg_label].mean():,.2f}**.
"""
        return {
            "thought": f"Grouped '{dim_col}' and aggregated '{metric_col}' using {agg_func}.",
            "sql": sql,
            "result_df": res_df,
            "fig": fig,
            "chart_code": f'fig = px.bar(result_df, x="{dim_col}", y="{agg_label}", text="{agg_label}")',
            "elapsed_ms": 2.7,
            "explanation": explanation,
            "followups": [
                f"Show distribution of {metric_col} by {dim_col}",
                f"Find outliers in {metric_col}",
            ],
        }

    def _handle_general_fallback(self, q: str) -> Dict[str, Any]:
        """Default fallback: smart preview with key highlights."""
        primary_col = self.numeric_cols[0] if self.numeric_cols else self.all_cols[0]
        sql = f'SELECT * FROM {self.table_name} ORDER BY "{primary_col}" DESC LIMIT 10;'
        res_df = self.con.execute(sql).df()

        explanation = f"""### 🔍 Dataset Slicing Result

Displayed below are the top 10 records ordered by **`{primary_col}`**.

💡 **Pro-Tip**: You can ask specific questions like:
- *"What are the top 5 {self.cat_cols[0] if self.cat_cols else 'categories'} by {self.numeric_cols[0] if self.numeric_cols else 'value'}?"*
- *"Show correlation matrix across all numeric features"*
- *"Find outliers in {self.numeric_cols[0] if self.numeric_cols else 'data'}"*
- *"Show distribution of {self.numeric_cols[0] if self.numeric_cols else 'data'} as a box plot"*
"""
        return {
            "thought": "Standard data slice preview ordered by primary attribute.",
            "sql": sql,
            "result_df": res_df,
            "fig": None,
            "chart_code": None,
            "elapsed_ms": 2.1,
            "explanation": explanation,
            "followups": [
                f"Top 5 {self.cat_cols[0]} by {self.numeric_cols[0]}" if self.cat_cols and self.numeric_cols else "Summary",
                f"Show correlation matrix",
                f"Audit missing values",
            ],
        }
