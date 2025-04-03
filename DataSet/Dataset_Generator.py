import pandas as pd
import numpy as np

np.random.seed(42)

num_rows = 10000

data = {
    "credit_score": np.random.randint(350, 901, num_rows),
    "years_with_bank": np.random.randint(0, 20, num_rows),
    "deposit_amount": np.random.randint(25000, 100000001, num_rows),
    "repayment_history": np.random.choice(["excellent", "good", "poor"], num_rows, p=[0.5, 0.3, 0.2]),
    "market_trend": np.random.choice(["rising", "stable", "falling"], num_rows, p=[0.5, 0.3, 0.2])
}

df = pd.DataFrame(data)

df["credit_score_adjustment"] = df["credit_score"].apply(lambda x: 
    np.random.randint(50, 101) if x > 750 else 
    np.random.randint(-100, -49) if x < 500 else 
    np.random.randint(-25, 26)
)

df["years_with_bank_adjustment"] = df["years_with_bank"].apply(lambda x: 
    np.random.randint(50, 101) if x >= 10 else 
    np.random.randint(25, 51) if x > 5 else 
    np.random.randint(-25, 26)
)

df["deposit_amount_adjustment"] = df["deposit_amount"].apply(lambda x: 
    np.random.randint(50, 101) if x > 10000000 else  
    np.random.randint(25, 51) if x > 100000 else 
    np.random.randint(-25, 26)
)

df["repayment_history_adjustment"] = df["repayment_history"].apply(lambda x: 
    np.random.randint(50, 101) if x == "excellent" else 
    np.random.randint(-100, -49) if x == "poor" else 
    np.random.randint(-25, 26)
)

df["market_trend_adjustment"] = df["market_trend"].apply(lambda x: 
    np.random.randint(25, 51) if x == "rising" else 
    np.random.randint(-50, -24) if x == "falling" else 
    np.random.randint(-25, 26)
)

df["final_interest_rate_adjustment_bps"] = (
    df["credit_score_adjustment"] +
    df["years_with_bank_adjustment"] +
    df["deposit_amount_adjustment"] +
    df["repayment_history_adjustment"] +
    df["market_trend_adjustment"]
)

df["final_interest_rate_adjustment_bps"] = df["final_interest_rate_adjustment_bps"].apply(
    lambda x: np.random.randint(-200, 301) if x < -200 or x > 300 else x
)

base_interest_rate = 7.5 / 100

df["adjusted_interest_rate"] = base_interest_rate + (df["final_interest_rate_adjustment_bps"] / 10000)

df["adjusted_interest_rate"] = df["adjusted_interest_rate"].apply(
    lambda x: x if x > 7.5 / 100 else 7.51 / 100 + np.random.uniform(0, 0.01)
)

df["adjusted_interest_rate"] = (df["adjusted_interest_rate"] * 100).round(2)

product_types = ["Home Loans"] * 5000 + ["Fixed Deposits"] * 5000
np.random.shuffle(product_types)
df["product_type"] = product_types

df.to_csv("credit_score_dataset_updated.csv", index=False)
