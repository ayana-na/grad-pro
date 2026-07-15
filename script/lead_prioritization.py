import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

print("="*60)
print(" Lead Prioritization System (with missing data handling)")
print("="*60)

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

model = joblib.load("models/conversion_model_optimized.pkl")
conversion_features = joblib.load("models/conversion_features_optimized.pkl")

query = """
SELECT
    l.lead_id,
    l.budget,
    l.engagement_score,
    l.urgency_level,
    l.source,
    l.price_gap,
    p.sqft,
    p.bedrooms,
    p.bathrooms,
    DATEDIFF(CURDATE(), p.listing_date) AS days_on_market,
    p.overall_qual,
    p.overall_cond,
    p.year_built,
    p.bsmt_sf,
    p.fireplaces,
    p.garage_cars,
    p.deck_sf,
    p.neighborhood
FROM leads l
JOIN properties p ON l.property_id = p.property_id
"""

df = pd.read_sql(query, engine)
print(f"Loaded {len(df)} leads")

numeric_cols = ['bsmt_sf', 'fireplaces', 'garage_cars', 'deck_sf', 'sqft', 'bedrooms', 'bathrooms',
                'days_on_market', 'overall_qual', 'overall_cond', 'year_built']
for col in numeric_cols:
    if col in df.columns:
        
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median() if df[col].dtype in ['float64','int64'] else 0, inplace=True)


df['urgency_level'] = df['urgency_level'].fillna('medium')
df['source'] = df['source'].fillna('Website')
df['neighborhood'] = df['neighborhood'].fillna(0)


df["price_per_sqft"] = df["budget"] / df["sqft"]
df["property_age"] = 2026 - df["year_built"]
df["quality_index"] = df["overall_qual"] * df["overall_cond"]
df["luxury_score"] = (df["garage_cars"] * 2) + (df["fireplaces"] * 1.5) + (df["deck_sf"] / 100)
df["price_to_income_ratio"] = df["budget"] / (df["engagement_score"] * 100000 + 1)
df["days_on_market_log"] = np.log1p(df["days_on_market"])
df["sqft_per_bedroom"] = df["sqft"] / (df["bedrooms"] + 1)
df["bath_per_bedroom"] = df["bathrooms"] / (df["bedrooms"] + 1)
df["has_basement"] = (df["bsmt_sf"] > 0).astype(int)
df["has_fireplace"] = (df["fireplaces"] > 0).astype(int)
df["has_garage"] = (df["garage_cars"] > 0).astype(int)


df_encoded = pd.get_dummies(df, columns=["urgency_level", "source", "neighborhood"], drop_first=True)


for col in conversion_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

X = df_encoded[conversion_features]
df["conversion_probability"] = model.predict_proba(X)[:, 1]

df["priority_score"] = (
    0.6 * df["conversion_probability"] +
    0.3 * df["engagement_score"] +
    0.1 * (df["price_gap"] > 0).astype(int)
)

df = df.sort_values("priority_score", ascending=False)
top_leads = df.head(10)

print("\n Top 10 Priority Leads:\n")
print(top_leads[["lead_id", "conversion_probability", "engagement_score", "priority_score"]])

top_leads.to_sql("lead_priorities", engine, if_exists="replace", index=False)

print("\n Lead priorities saved to database")