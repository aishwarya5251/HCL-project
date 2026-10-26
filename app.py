"""
DataSense AI — Next-Gen Conversational Data Analysis Assistant
100% Local, Zero-API-Key Required. Powered by DuckDB, Plotly, Pandas, and Semantic NLP.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Add current folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.data_loader import load_csv, get_dataset_summary
from engine.local_nlp_engine import LocalNLPEngine
from engine.query_executor import QueryExecutor
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="DataSense AI — Local Conversational Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Modern Styling
st.markdown(
    """
<style>
    /* Global layout enhancements */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 0px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .kpi-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 500;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 14px;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .insight-box {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #15803D;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_dataset_name" not in st.session_state:
    st.session_state.active_dataset_name = "E-Commerce Sales"

if "chart_theme" not in st.session_state:
    st.session_state.chart_theme = "Viridis"

if "df" not in st.session_state:
    default_sample = os.path.join(os.path.dirname(__file__), "samples", "ecommerce_sales.csv")
    if os.path.exists(default_sample):
        st.session_state.df = load_csv(default_sample)
    else:
        st.session_state.df = None


# Sidebar: Data Hub & Settings
with st.sidebar:
    st.markdown("### ⚡ **DataSense Engine**")
    st.caption("100% Local Analytical Engine • Zero API Key • Private & Instant")

    st.markdown("---")
    st.subheader("📁 Dataset Manager")

    data_source = st.radio(
        "Source",
        ["Curated Datasets", "Upload CSV"],
        horizontal=True,
    )

    sample_files = {
        "E-Commerce Sales": os.path.join(os.path.dirname(__file__), "samples", "ecommerce_sales.csv"),
        "HR Analytics & Attrition": os.path.join(os.path.dirname(__file__), "samples", "hr_analytics.csv"),
        "SaaS Customer Churn": os.path.join(os.path.dirname(__file__), "samples", "saas_churn.csv"),
    }

    if data_source == "Curated Datasets":
        chosen_sample = st.selectbox(
            "Select Dataset",
            list(sample_files.keys()),
            index=list(sample_files.keys()).index(st.session_state.active_dataset_name)
            if st.session_state.active_dataset_name in sample_files
            else 0,
        )
        if st.session_state.active_dataset_name != chosen_sample or st.session_state.df is None:
            st.session_state.active_dataset_name = chosen_sample
            st.session_state.df = load_csv(sample_files[chosen_sample])
            st.session_state.messages = []
            st.rerun()

    else:
        uploaded_file = st.file_uploader(
            "Upload any CSV dataset",
            type=["csv"],
            help="Upload your personal or business CSV files.",
        )
        if uploaded_file is not None:
            if st.session_state.active_dataset_name != uploaded_file.name or st.session_state.df is None:
                st.session_state.active_dataset_name = uploaded_file.name
                st.session_state.df = load_csv(uploaded_file)
                st.session_state.messages = []
                st.success(f"Loaded '{uploaded_file.name}'!")
                st.rerun()

    # Color Theme Preference
    st.markdown("---")
    st.subheader("🎨 Chart Appearance")
    st.session_state.chart_theme = st.selectbox(
        "Color Palette",
        ["Viridis", "Blues", "Teal", "Turbo", "Plasma", "Plotly"],
        index=0,
    )

    # Conversation Management
    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Export Report
    if len(st.session_state.messages) > 0:
        report_md = f"# DataSense AI — Executive Analysis Report\n\n"
        report_md += f"**Dataset**: `{st.session_state.active_dataset_name}`  \n"
        report_md += f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n---\n\n"
        for idx, m in enumerate(st.session_state.messages):
            role_label = "👤 User Query" if m["role"] == "user" else "⚡ DataSense Insight"
            report_md += f"### {role_label}\n{m['content']}\n\n"
            if m.get("sql"):
                report_md += f"```sql\n{m['sql']}\n```\n\n"
        st.download_button(
            "📥 Download Executive Report (.md)",
            data=report_md,
            file_name=f"executive_report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# Main Content Header
st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">⚡ DataSense AI — Local Conversational Analyst</div>
    <div class="hero-sub">Deep Natural-Language Q&A, Automatic DuckDB SQL, Statistical Profiles, and Interactive Plotly Visuals — 100% Offline.</div>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.df is None:
    st.info("👋 Select a dataset or upload a CSV file from the sidebar to begin.")
    st.stop()

# Instantiate Engines
df_active = st.session_state.df
summary = get_dataset_summary(df_active)
nlp_engine = LocalNLPEngine(df_active, table_name="dataset")
sql_executor = QueryExecutor(df_active, table_name="dataset")

# Top KPI Overview Bar
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-val">{summary['rows']:,}</div>
            <div class="kpi-sub">Rows Analyzed</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Features / Columns</div>
            <div class="kpi-val">{summary['columns']}</div>
            <div class="kpi-sub">{len(summary['numeric_cols'])} Num • {len(summary['categorical_cols'])} Cat</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    clean_pct = round(100 - summary["missing_pct"], 1)
    health_color = "#10B981" if clean_pct >= 95 else "#F59E0B"
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Data Completeness</div>
            <div class="kpi-val" style="color: {health_color}">{clean_pct}%</div>
            <div class="kpi-sub">{summary['missing_cells']} Missing Cells</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">In-Memory Engine</div>
            <div class="kpi-val" style="color: #6366F1">DuckDB</div>
            <div class="kpi-sub">Sub-millisecond latency</div>
        </div>""",
        unsafe_allow_html=True,
    )


# Main Tabs Navigation
tab_chat, tab_eda, tab_studio, tab_sql, tab_data = st.tabs(
    [
        "💬 Conversational Analyst",
        "📊 Automated EDA & Diagnostics",
        "🎨 Visual Chart Studio",
        "🛠️ DuckDB SQL Lab",
        "📋 Data Dictionary & Table",
    ]
)


# ==========================================
# TAB 1: Conversational Analyst
# ==========================================
with tab_chat:
    # Dataset-Specific Suggested Prompts
    if len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Suggested Questions for this Dataset:")
        suggestions_map = {
            "E-Commerce Sales": [
                "What are the top 5 product sub-categories by sales?",
                "Show monthly sales trend over time",
                "Is there correlation between discount and profit?",
                "Detect outliers in profit",
                "Share of sales by region",
            ],
            "HR Analytics & Attrition": [
                "What is the average annual salary by department?",
                "Show distribution of annual salaries as a box plot",
                "Breakdown of employee count by job role",
                "Find outliers in annual salary",
                "Summary statistics",
            ],
            "SaaS Customer Churn": [
                "Compare churn count across contract types",
                "Show distribution of monthly charges",
                "Is there correlation between tenure months and total charges?",
                "Detect outliers in total charges",
                "Customer proportion breakdown by subscription tier",
            ],
        }

        active_pills = suggestions_map.get(
            st.session_state.active_dataset_name,
            [
                f"What are the top 5 {summary['categorical_cols'][0] if summary['categorical_cols'] else 'items'}?",
                f"Show distribution of {summary['numeric_cols'][0] if summary['numeric_cols'] else 'data'}",
                "Show correlation matrix across numeric columns",
                "Detect outliers and anomalies",
                "Audit missing and null values",
            ],
        )

        p_cols = st.columns(len(active_pills))
        selected_prompt = None
        for i, p in enumerate(active_pills):
            if p_cols[i].button(p, key=f"quick_pill_{i}", use_container_width=True):
                selected_prompt = p

        if selected_prompt:
            st.session_state.messages.append({"role": "user", "content": selected_prompt})
            st.rerun()

    # Render Conversation Messages
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])

            # SQL Accordion
            if msg.get("sql"):
                with st.expander(f"🔍 DuckDB SQL Query ({msg.get('elapsed_ms', 0)} ms)", expanded=False):
                    st.code(msg["sql"], language="sql")

            # Data Table Accordion
            if msg.get("result_df") is not None and not msg["result_df"].empty:
                with st.expander(
                    f"📄 Result Data Slice ({len(msg['result_df'])} rows × {len(msg['result_df'].columns)} cols)",
                    expanded=False,
                ):
                    st.dataframe(msg["result_df"], use_container_width=True)
                    csv_export = msg["result_df"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download CSV Slice",
                        data=csv_export,
                        file_name=f"datasense_slice_{idx}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{idx}",
                    )

            # Plotly Visualization
            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], use_container_width=True, key=f"chat_fig_{idx}")

            # Follow-up Suggestions
            if msg.get("followups"):
                st.markdown("**Suggested Follow-ups:**")
                f_cols = st.columns(len(msg["followups"]))
                for f_idx, followup in enumerate(msg["followups"]):
                    if f_cols[f_idx].button(f"👉 {followup}", key=f"btn_fu_{idx}_{f_idx}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": followup})
                        st.rerun()

    # Chat Input Box
    user_query = st.chat_input("Ask any analytical question, request charts, or search for anomalies...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Analyzing dataset with DuckDB..."):
                t_start = time.perf_counter()
                res = nlp_engine.query(user_query)
                elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

                explanation = res.get("explanation", "")
                sql = res.get("sql")
                result_df = res.get("result_df")
                fig = res.get("fig")
                followups = res.get("followups", [])

                # Render response
                st.markdown(explanation)

                if sql:
                    with st.expander(f"🔍 DuckDB SQL Query ({elapsed_ms} ms)", expanded=False):
                        st.code(sql, language="sql")

                if result_df is not None and not result_df.empty:
                    with st.expander(
                        f"📄 Result Data Slice ({len(result_df)} rows × {len(result_df.columns)} cols)",
                        expanded=False,
                    ):
                        st.dataframe(result_df, use_container_width=True)

                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                if followups:
                    st.markdown("**Suggested Follow-ups:**")
                    f_cols = st.columns(len(followups))
                    for f_idx, f_text in enumerate(followups):
                        f_cols[f_idx].caption(f"👉 {f_text}")

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": explanation,
                        "sql": sql,
                        "result_df": result_df,
                        "fig": fig,
                        "elapsed_ms": elapsed_ms,
                        "followups": followups,
                    }
                )


# ==========================================
# TAB 2: Automated EDA & Diagnostics
# ==========================================
with tab_eda:
    st.subheader("📊 Automated Exploratory Data Analysis & Diagnostics")
    st.caption("Comprehensive data health diagnostics, distributions, correlation patterns, and anomaly scans.")

    eda_sub_1, eda_sub_2, eda_sub_3, eda_sub_4 = st.tabs(
        [
            "📈 Distributions",
            "🔗 Correlation Heatmap",
            "🚨 Anomaly & Outlier Scanner",
            "🏷️ Categorical Profiles",
        ]
    )

    with eda_sub_1:
        if summary["numeric_cols"]:
            selected_num = st.selectbox("Select Numeric Column to Profile", summary["numeric_cols"], key="eda_num_sel")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                fig_dist = px.histogram(
                    df_active,
                    x=selected_num,
                    marginal="box",
                    nbins=30,
                    title=f"Histogram & Boxplot: {selected_num}",
                    color_discrete_sequence=["#3B82F6"],
                )
                fig_dist.update_layout(template="plotly_white")
                st.plotly_chart(fig_dist, use_container_width=True)
            with col_b:
                s_ser = df_active[selected_num].dropna()
                st.markdown(f"**Key Statistics: `{selected_num}`**")
                stat_dict = {
                    "Mean": f"{s_ser.mean():,.2f}",
                    "Std Dev": f"{s_ser.std():,.2f}",
                    "Min": f"{s_ser.min():,.2f}",
                    "25th Percentile": f"{s_ser.quantile(0.25):,.2f}",
                    "Median (50th)": f"{s_ser.median():,.2f}",
                    "75th Percentile": f"{s_ser.quantile(0.75):,.2f}",
                    "Max": f"{s_ser.max():,.2f}",
                    "Skewness": f"{s_ser.skew():+0.2f}",
                }
                st.table(pd.DataFrame(stat_dict.items(), columns=["Metric", "Value"]))
        else:
            st.info("No numeric columns detected in this dataset.")

    with eda_sub_2:
        if len(summary["numeric_cols"]) >= 2:
            corr_matrix = df_active[summary["numeric_cols"]].corr().round(3)
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix (Pearson)",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            fig_corr.update_layout(template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("At least two numeric columns are needed to calculate correlations.")

    with eda_sub_3:
        if summary["numeric_cols"]:
            outlier_col = st.selectbox("Select Feature for Anomaly Scan", summary["numeric_cols"], key="outlier_sel")
            s_out = df_active[outlier_col].dropna()
            q25 = s_out.quantile(0.25)
            q75 = s_out.quantile(0.75)
            iqr = q75 - q25
            lb = q25 - 1.5 * iqr
            ub = q75 + 1.5 * iqr
            outliers = df_active[(df_active[outlier_col] < lb) | (df_active[outlier_col] > ub)]

            st.markdown(
                f"""
            - **Lower Boundary**: `{lb:,.2f}` | **Upper Boundary**: `{ub:,.2f}`
            - **Anomalies Detected**: **{len(outliers)}** out of {len(df_active)} records ({round(len(outliers)/len(df_active)*100, 2)}%)
            """
            )
            if not outliers.empty:
                st.dataframe(outliers, use_container_width=True)
            else:
                st.success("No anomalies detected beyond the 1.5 × IQR boundary!")
        else:
            st.info("No numeric columns available.")

    with eda_sub_4:
        if summary["categorical_cols"]:
            cat_choice = st.selectbox("Select Categorical Feature", summary["categorical_cols"], key="eda_cat_sel")
            top_counts = df_active[cat_choice].value_counts().reset_index()
            top_counts.columns = [cat_choice, "Frequency"]
            fig_cat = px.bar(
                top_counts.head(15),
                x=cat_choice,
                y="Frequency",
                text="Frequency",
                title=f"Category Frequency: {cat_choice} (Top 15)",
                color="Frequency",
                color_continuous_scale="Teal",
            )
            fig_cat.update_layout(template="plotly_white")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No categorical columns detected.")


# ==========================================
# TAB 3: Visual Chart Studio
# ==========================================
with tab_studio:
    st.subheader("🎨 Custom Visual Chart Studio")
    st.caption("Design, configure, and download interactive visualizations from your dataset.")

    cs_c1, cs_c2, cs_c3, cs_c4 = st.columns(4)
    with cs_c1:
        c_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box", "Violin", "Pie", "Histogram"])
    with cs_c2:
        x_axis = st.selectbox("X-Axis Feature", df_active.columns)
    with cs_c3:
        y_axis = st.selectbox("Y-Axis Feature", [c for c in df_active.columns if c != x_axis] + [x_axis])
    with cs_c4:
        color_by = st.selectbox("Group / Color By", ["(None)"] + df_active.columns.tolist())

    color_col = None if color_by == "(None)" else color_by

    try:
        if c_type == "Bar":
            fig_custom = px.bar(df_active, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}")
        elif c_type == "Line":
            fig_custom = px.line(df_active, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} Trend over {x_axis}")
        elif c_type == "Scatter":
            fig_custom = px.scatter(df_active, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
        elif c_type == "Box":
            fig_custom = px.box(df_active, x=x_axis, y=y_axis, color=color_col, title=f"Distribution of {y_axis} by {x_axis}")
        elif c_type == "Violin":
            fig_custom = px.violin(df_active, x=x_axis, y=y_axis, color=color_col, box=True, title=f"Violin Plot of {y_axis} by {x_axis}")
        elif c_type == "Pie":
            fig_custom = px.pie(df_active, names=x_axis, values=y_axis if y_axis in summary["numeric_cols"] else None, title=f"Composition of {x_axis}")
        elif c_type == "Histogram":
            fig_custom = px.histogram(df_active, x=x_axis, color=color_col, title=f"Histogram of {x_axis}")

        fig_custom.update_layout(template="plotly_white")
        st.plotly_chart(fig_custom, use_container_width=True)
    except Exception as e_chart:
        st.error(f"Could not render chart: {e_chart}")


# ==========================================
# TAB 4: DuckDB SQL Lab
# ==========================================
with tab_sql:
    st.subheader("🛠️ DuckDB SQL Lab")
    st.caption("Query the in-memory dataset table (`dataset`) using standard ANSI SQL.")

    sql_template = f"SELECT * FROM dataset LIMIT 10;"
    sql_query_input = st.text_area("SQL Statement", value=sql_template, height=130)

    col_run, col_time = st.columns([1, 4])
    if col_run.button("🚀 Execute SQL", type="primary", use_container_width=True):
        t_sql_start = time.perf_counter()
        exec_out = sql_executor.execute_sql(sql_query_input)
        sql_ms = round((time.perf_counter() - t_sql_start) * 1000, 2)

        if exec_out["success"]:
            st.success(f"Executed in {sql_ms} ms ({exec_out['rows']} rows returned)")
            st.dataframe(exec_out["data"], use_container_width=True)
            csv_sql = exec_out["data"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Query Output (CSV)",
                data=csv_sql,
                file_name=f"sql_output_{int(time.time())}.csv",
                mime="text/csv",
            )
        else:
            st.error(f"SQL Error: {exec_out['error']}")


# ==========================================
# TAB 5: Data Dictionary & Table
# ==========================================
with tab_data:
    st.subheader(f"📋 Dataset Schema & Profiles: {st.session_state.active_dataset_name}")

    st.markdown("### 🔍 Raw Table Inspector")
    st.dataframe(df_active, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Column Data Dictionary")
    dict_rows = []
    for c in summary["columns_meta"]:
        dict_rows.append(
            {
                "Column Name": c["name"],
                "Data Type": c["dtype"],
                "Null Values": f"{c['null_count']} ({c['null_pct']}%)",
                "Unique Distinct": c["unique_count"],
                "Sample Values": ", ".join(map(str, c["sample_values"][:3])),
            }
        )
    st.table(pd.DataFrame(dict_rows))

    st.markdown("---")
    st.markdown("### 📊 Descriptive Statistics (`describe()`)")
    st.dataframe(df_active.describe(include="all").transpose(), use_container_width=True)
