import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*60)
print(" Advanced Sales Forecasting System")
print("="*60)


engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")


query = """
SELECT
DATE_FORMAT(created_at,'%%Y-%%m') as month,
COUNT(*) as total_leads,
SUM(CASE WHEN conversion_status = 1 THEN 1 ELSE 0 END) as conversions,
SUM(CASE WHEN conversion_status = 1 THEN budget ELSE 0 END) as revenue
FROM leads
GROUP BY month
ORDER BY month
"""

df = pd.read_sql(query, engine)

if df.empty:
    logger.error("No data found")
    exit()


df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month")
df.set_index("month", inplace=True)


df["conversion_rate"] = df["conversions"] / df["total_leads"]

print("\n Historical Data")
print(df.tail())


X = np.arange(len(df)).reshape(-1,1)



lead_model = LinearRegression()
lead_model.fit(X, df["total_leads"])

lead_pred = lead_model.predict(X)

lead_mae = mean_absolute_error(df["total_leads"], lead_pred)



rate_model = LinearRegression()
rate_model.fit(X, df["conversion_rate"])

rate_pred = rate_model.predict(X)

rate_mape = mean_absolute_percentage_error(df["conversion_rate"], rate_pred)

print("\n Model Performance")
print(f"Lead MAE: {lead_mae:.2f}")
print(f"Conversion Rate MAPE: {rate_mape*100:.1f}%")



future_months = 6

future_X = np.arange(len(df), len(df)+future_months).reshape(-1,1)

future_leads = lead_model.predict(future_X)
future_rates = rate_model.predict(future_X)

future_leads = np.maximum(0, future_leads).astype(int)
future_rates = np.clip(future_rates, 0, 0.75)

future_conversions = (future_leads * future_rates).astype(int)


avg_revenue = df["revenue"].sum() / df["conversions"].sum()

future_revenue = future_conversions * avg_revenue


last_month = df.index[-1]

future_dates = pd.date_range(
start=last_month + pd.offsets.MonthBegin(),
periods=future_months,
freq="MS"
)

forecast_df = pd.DataFrame({
"month":future_dates,
"predicted_leads":future_leads,
"predicted_conversion_rate":future_rates,
"predicted_conversions":future_conversions,
"predicted_revenue":future_revenue
})

print("\n Forecast Results")
print(forecast_df)


forecast_df.to_sql(
"sales_forecast",
engine,
if_exists="replace",
index=False
)

print("\n Forecast saved to database")



fig = go.Figure()

fig.add_trace(go.Scatter(
x=df.index,
y=df["conversions"],
mode="lines+markers",
name="Historical Conversions"
))

fig.add_trace(go.Scatter(
x=future_dates,
y=future_conversions,
mode="lines+markers",
name="Forecast Conversions",
line=dict(dash="dash")
))

fig.update_layout(
title="Sales Conversion Forecast",
xaxis_title="Month",
yaxis_title="Conversions"
)

fig.write_html("sales_forecast.html")

print("\n Chart saved as sales_forecast.html")

print("="*60)
print(" Sales Forecasting Completed")
print("="*60)