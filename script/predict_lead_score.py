import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime

print("=== Predict Lead Scores (Ultra Fast) ===")

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

model = joblib.load("models/lead_model.pkl")
columns = joblib.load("models/lead_model_columns.pkl")

df = pd.read_sql("SELECT lead_id, budget, urgency_level, source, engagement_score, price_gap FROM leads WHERE lead_category IS NULL", engine)

if df.empty:
    print("No leads need scoring.")
    exit()

print(f"Scoring {len(df)} leads...")

df_dummies = pd.get_dummies(df, columns=["urgency_level", "source"]).reindex(columns=columns, fill_value=0)
proba = model.predict_proba(df_dummies)[:, 1]
categories = pd.cut(proba*100, bins=[0,40,70,100], labels=["COLD","WARM","HOT"], include_lowest=True)

update_data = pd.DataFrame({
    "lead_id": df["lead_id"],
    "conversion_probability": proba.round(3),
    "lead_category": categories,
    "updated_at": datetime.now()
})

print("Bulk updating (very fast)...")
with engine.begin() as conn:
    update_data.to_sql("temp_updates", conn, index=False)
    conn.execute(text("""
        UPDATE leads l
        JOIN temp_updates t ON l.lead_id = t.lead_id
        SET l.conversion_probability = t.conversion_probability,
            l.lead_category = t.lead_category,
            l.updated_at = t.updated_at
    """))
    conn.execute(text("DROP TABLE IF EXISTS temp_updates"))

print(f" Done! Updated {len(df)} leads.")