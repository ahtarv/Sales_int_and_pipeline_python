import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

os.makedirs("outputs/figures", exist_ok=True)

sales_data = pd.read_csv("data/sales_data.csv", parse_dates=["date"])

# Monthly Revenue
monthly_revenue = (
    sales_data
    .assign(month=sales_data["date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month", as_index=False)["revenue"]
    .sum()
    .rename(columns={"revenue": "total_revenue"})
    .sort_values("month")
)

# Build time series
ts = monthly_revenue.set_index("month")["total_revenue"]
ts.index.freq = "MS"  # month start frequency

# Decompose (additive)
decomp = seasonal_decompose(ts, model="additive")

fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig("outputs/figures/decomposition.png", dpi=100)
plt.close()
print("Saved: outputs/figures/decomposition.png")