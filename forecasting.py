import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima  # pip install pmdarima

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

ts = monthly_revenue.set_index("month")["total_revenue"]
ts.index.freq = "MS"

# Auto ARIMA (same as R's auto.arima with seasonal=TRUE)
arima_model = auto_arima(ts, seasonal=True, m=12, suppress_warnings=True, stepwise=True)
print(arima_model.summary())

# Forecast 6 months
arima_forecast = arima_model.predict(n_periods=6, return_conf_int=True)
forecast_vals, conf_int = arima_forecast

# Naive forecast (last known value repeated)
naive_forecast = [ts.iloc[-1]] * 6

# Build future date index
future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=6, freq="MS")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ts.index, ts.values, label="Actual", color="black")
plt.plot(future_dates, forecast_vals, label="Seasonal ARIMA Forecast", color="blue")
plt.fill_between(future_dates, conf_int[:, 0], conf_int[:, 1], alpha=0.2, color="blue")
plt.plot(future_dates, naive_forecast, label="Naive Forecast", color="red", linestyle="--")
plt.title("6-Month Revenue Forecast (Seasonal ARIMA)")
plt.xlabel("Time")
plt.ylabel("Revenue")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("outputs/figures/revenue_forecast.png", dpi=100)
plt.show()

print("\nForecast values:")
for date, val in zip(future_dates, forecast_vals):
    print(f"  {date.strftime('%Y-%m')}: {val:,.2f}")