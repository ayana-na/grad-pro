import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("="*60)
print(" Customer Segmentation System")
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


seg_features = [
    "budget",
    "engagement_score",
    "price_gap",
    "conversion_probability"
]

X_seg = df[seg_features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_seg)


kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)


cluster_stats = df.groupby("cluster")["conversion_probability"].mean()
cluster_order = cluster_stats.sort_values(ascending=False).index

cluster_labels = {
    cluster_order[0]: "Hot Buyers",
    cluster_order[1]: "Serious Buyers",
    cluster_order[2]: "Casual Browsers"
}

df["segment"] = df["cluster"].map(cluster_labels)

print("\n Segment Distribution\n")
print(df["segment"].value_counts())

print("\n Sample Hot Buyers\n")
print(df[df["segment"] == "Hot Buyers"].head()[[
    "lead_id",
    "budget",
    "conversion_probability",
    "engagement_score"
]])


df[[
    "lead_id",
    "budget",
    "conversion_probability",
    "segment"
]].to_sql(
    "customer_segments",
    engine,
    if_exists="replace",
    index=False
)

print("\n Customer segments saved to database")
print("="*60)
print("Segmentation Completed")
print("="*60)