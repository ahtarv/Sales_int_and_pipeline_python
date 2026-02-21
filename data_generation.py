import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

num_days = 730
dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")

products = pd.DataFrame({
    "product_id": [f"P{i}" for i in range(1, 13)],
    "category": ["Electronics"] * 4 + ["Clothing"] * 4 + ["Home"] * 4,
    "base_price": (
        list(np.random.uniform(200, 600, 4)) +
        list(np.random.uniform(20, 60, 4)) +
        list(np.random.uniform(50, 150, 4))
    )
})

regions = ["North", "South", "East", "West"]
channels = ["Online", "Store"]

def generate_daily_sales(date):
    product = products.sample(1).iloc[0]

    month_factor = 1.4 if date.strftime("%m") in ["11", "12"] else 1
    weekday_factor = 1.2 if date.day_name() in ["Saturday", "Sunday"] else 1

    promotion = np.random.choice(["Yes", "No"], p=[0.25, 0.75])
    promo_factor = 1.5 if promotion == "Yes" else 1

    quantity = np.random.poisson(lam=5 * month_factor * weekday_factor * promo_factor)
    unit_price = round(product["base_price"] * (0.85 if promotion == "Yes" else 1), 2)

    return {
        "date": date.date(),
        "order_id": f"ORD{random.randint(100000, 999999)}",
        "customer_id": f"C{random.randint(1, 500)}",
        "product_id": product["product_id"],
        "category": product["category"],
        "unit_price": unit_price,
        "quantity": quantity,
        "revenue": round(unit_price * quantity, 2),
        "region": random.choice(regions),
        "sales_channel": np.random.choice(channels, p=[0.6, 0.4]),
        "promotion": promotion
    }

sales_data = pd.DataFrame([generate_daily_sales(d) for d in dates])
print(sales_data.head())
sales_data.to_csv("data/sales_data.csv", index=False)