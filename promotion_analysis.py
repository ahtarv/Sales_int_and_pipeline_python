import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sales_data = pd.read_csv("data/sales_data.csv", parse_dates=["date"])

# Promotion Summary
promo_summary = sales_data.groupby("promotion").agg(
    avg_revenue=("revenue", "mean"),
    median_revenue=("revenue", "median"),
    total_revenue=("revenue", "sum"),
    avg_quantity=("quantity", "mean"),
    orders=("revenue", "count")
).reset_index()
print(promo_summary)

# Plot 1: Revenue Distribution - Promo vs No Promo (Boxplot)
promo_groups = [
    sales_data[sales_data["promotion"] == "Yes"]["revenue"].values,
    sales_data[sales_data["promotion"] == "No"]["revenue"].values
]
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot(promo_groups, labels=["Yes", "No"], patch_artist=True, notch=False)
bp["boxes"][0].set_facecolor("steelblue")
bp["boxes"][1].set_facecolor("salmon")
ax.set_title("Revenue Distribution: Promotion vs No Promotion")
ax.set_xlabel("Promotion")
ax.set_ylabel("Revenue")
plt.tight_layout()
plt.show()

# Category + Promotion Summary
category_promo = sales_data.groupby(["category", "promotion"]).agg(
    avg_revenue=("revenue", "mean"),
    total_revenue=("revenue", "sum")
).reset_index()
print(category_promo)

# Plot 2: Avg Revenue by Category and Promotion (Grouped Bar)
categories = category_promo["category"].unique()
promo_vals = category_promo["promotion"].unique()
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
for i, promo in enumerate(promo_vals):
    subset = category_promo[category_promo["promotion"] == promo]
    ax.bar(x + i * width, subset["avg_revenue"], width, label=promo)

ax.set_title("Average Revenue by Category and Promotion")
ax.set_xlabel("Category")
ax.set_ylabel("Average Revenue")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(categories)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/promotion_category_revenue.png", dpi=100)
plt.show()

# Quantity vs Revenue by Promotion
qty_rev = sales_data.groupby("promotion").agg(
    avg_quantity=("quantity", "mean"),
    avg_revenue=("revenue", "mean")
)
print(qty_rev)