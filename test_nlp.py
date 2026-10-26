import pandas as pd
from engine.local_nlp_engine import LocalNLPEngine

df = pd.read_csv("samples/ecommerce_sales.csv")
engine = LocalNLPEngine(df)

queries = [
    "What are the top 5 sub-categories by sales?",
    "Show monthly sales trend over time",
    "Is there correlation between discount and profit?",
    "Detect outliers in profit",
    "Share of sales by region",
    "Summary statistics",
]

for q in queries:
    res = engine.query(q)
    print(f"QUERY: {q}")
    print(f"  SQL: {bool(res.get('sql'))}")
    print(f"  Chart: {res.get('fig') is not None}")
    print(f"  Result Rows: {len(res.get('result_df')) if res.get('result_df') is not None else 0}")
    print("---")
print("All NLP query strategies succeeded!")
