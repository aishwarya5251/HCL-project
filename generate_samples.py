import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

os.makedirs("samples", exist_ok=True)
random.seed(42)
np.random.seed(42)

# 1. E-Commerce Sales Dataset
n_orders = 200
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=random.randint(0, 500)) for _ in range(n_orders)]
regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
categories = {
    "Technology": ["Laptops", "Smartphones", "Accessories", "Printers"],
    "Furniture": ["Chairs", "Desks", "Bookcases", "Tables"],
    "Office Supplies": ["Paper", "Binders", "Storage", "Pens"]
}
segments = ["Consumer", "Corporate", "Small Business"]

records_sales = []
for i in range(1, n_orders + 1):
    order_date = random.choice(dates)
    region = random.choice(regions)
    cat = random.choice(list(categories.keys()))
    subcat = random.choice(categories[cat])
    seg = random.choice(segments)
    qty = random.randint(1, 10)
    base_price = {"Technology": 450, "Furniture": 220, "Office Supplies": 45}[cat]
    discount = round(random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3]), 2)
    sales = round(qty * base_price * (1 + np.random.uniform(-0.25, 0.35)) * (1 - discount), 2)
    cost = round(sales * np.random.uniform(0.55, 0.85), 2)
    profit = round(sales - cost, 2)
    
    records_sales.append({
        "Order_ID": f"ORD-{2023000 + i}",
        "Date": order_date.strftime("%Y-%m-%d"),
        "Region": region,
        "Customer_Segment": seg,
        "Category": cat,
        "Sub_Category": subcat,
        "Quantity": qty,
        "Discount": discount,
        "Sales": sales,
        "Profit": profit,
        "Margin_Pct": round((profit / sales) * 100, 1) if sales > 0 else 0.0
    })

df_sales = pd.DataFrame(records_sales)
df_sales.to_csv("samples/ecommerce_sales.csv", index=False)
print("Created samples/ecommerce_sales.csv")

# 2. HR Analytics Dataset
n_emp = 150
depts = ["Engineering", "Sales", "Marketing", "Human Resources", "Product"]
roles = {
    "Engineering": ["Software Engineer", "Senior Dev", "QA Engineer", "DevOps"],
    "Sales": ["Sales Exec", "Account Manager", "Sales Lead"],
    "Marketing": ["Marketing Specialist", "Content Strategist", "SEO Lead"],
    "Human Resources": ["HR Generalist", "Recruiter", "HR Manager"],
    "Product": ["Product Manager", "UX Designer", "Product Analyst"]
}

records_hr = []
for i in range(1, n_emp + 1):
    dept = random.choice(depts)
    role = random.choice(roles[dept])
    age = random.randint(22, 58)
    years = min(random.randint(1, 15), age - 21)
    base_sal = {"Engineering": 95000, "Product": 105000, "Sales": 75000, "Marketing": 68000, "Human Resources": 62000}[dept]
    salary = int(base_sal + (years * 3200) + np.random.normal(0, 7000))
    satisfaction = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.35, 0.25, 0.15])[0]
    perf_rating = random.choices(["Needs Improvement", "Meets Expectations", "Exceeds Expectations", "Outstanding"], weights=[0.1, 0.5, 0.3, 0.1])[0]
    attrition_prob = 0.4 if satisfaction <= 2 else (0.1 if years > 5 else 0.2)
    attrition = "Yes" if random.random() < attrition_prob else "No"
    
    records_hr.append({
        "Employee_ID": f"EMP-{1000 + i}",
        "Age": age,
        "Department": dept,
        "Job_Role": role,
        "Years_At_Company": years,
        "Annual_Salary": salary,
        "Job_Satisfaction": satisfaction,
        "Performance_Rating": perf_rating,
        "Attrition": attrition
    })

df_hr = pd.DataFrame(records_hr)
df_hr.to_csv("samples/hr_analytics.csv", index=False)
print("Created samples/hr_analytics.csv")

# 3. SaaS Customer Churn Dataset
n_cust = 180
tiers = ["Basic", "Professional", "Enterprise"]
contracts = ["Month-to-month", "One year", "Two year"]

records_churn = []
for i in range(1, n_cust + 1):
    tier = random.choice(tiers)
    contract = random.choice(contracts)
    tenure = random.randint(1, 48)
    tier_fee = {"Basic": 29, "Professional": 99, "Enterprise": 299}[tier]
    monthly = round(tier_fee + np.random.uniform(-5, 20), 2)
    total = round(monthly * tenure, 2)
    support_tickets = random.choices([0, 1, 2, 3, 4, 7, 10], weights=[0.3, 0.25, 0.2, 0.12, 0.08, 0.03, 0.02])[0]
    churn_prob = 0.55 if contract == "Month-to-month" and support_tickets >= 3 else 0.12
    churn = "Yes" if random.random() < churn_prob else "No"
    
    records_churn.append({
        "Customer_ID": f"CUST-{5000 + i}",
        "Subscription_Tier": tier,
        "Contract_Type": contract,
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly,
        "Total_Charges": total,
        "Support_Tickets": support_tickets,
        "Churn": churn
    })

df_churn = pd.DataFrame(records_churn)
df_churn.to_csv("samples/saas_churn.csv", index=False)
print("Created samples/saas_churn.csv")
