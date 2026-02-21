import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for saving files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.makedirs("outputs/figures", exist_ok=True)

sales_data = pd.read_csv("data/sales_data.csv", parse_dates=["date"])

print(sales_data.dtypes)
print(sales_data.describe())

# Daily Revenue Over Time
plt.figure(figsize=(10, 5))
plt.plot(sales_data["date"], sales_data["revenue"], alpha=0.6, color="steelblue")
plt.title("Daily Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("outputs/figures/daily_revenue.png", dpi=100)
plt.close()
print("Saved: outputs/figures/daily_revenue.png")

# Monthly Revenue Trend
monthly_revenue = (
    sales_data
    .assign(month=sales_data["date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month", as_index=False)["revenue"]
    .sum()
    .rename(columns={"revenue": "total_revenue"})
)

plt.figure(figsize=(10, 5))
plt.plot(monthly_revenue["month"], monthly_revenue["total_revenue"], linewidth=1.1, color="darkgreen")
plt.scatter(monthly_revenue["month"], monthly_revenue["total_revenue"], color="darkgreen", s=30)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.savefig("outputs/figures/monthly_revenue_trend.png", dpi=100)
plt.close()
print("Saved: outputs/figures/monthly_revenue_trend.png")

# Revenue Distribution by Category (Boxplot)
categories = sales_data["category"].unique()
data_by_cat = [sales_data[sales_data["category"] == c]["revenue"].values for c in categories]

plt.figure(figsize=(8, 5))
plt.boxplot(data_by_cat, labels=categories, patch_artist=True)
plt.title("Revenue Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("outputs/figures/revenue_by_category.png", dpi=100)
plt.close()
print("Saved: outputs/figures/revenue_by_category.png")

# Promotion Summary
promo_summary = sales_data.groupby("promotion")["revenue"].agg(
    avg_revenue="mean",
    median_revenue="median",
    total_revenue="sum"
)
print(promo_summary)