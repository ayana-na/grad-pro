import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib

print("Connecting to database...")

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

# --------------------------------------------------
# اختيار Lead
# --------------------------------------------------

lead_query = """
SELECT lead_id,budget,engagement_score
FROM leads
WHERE lead_category IN ('HOT','WARM')
ORDER BY RAND()
LIMIT 1
"""

lead = pd.read_sql(lead_query,engine)

if lead.empty:
    print("No leads found")
    exit()

lead_id = lead.iloc[0]["lead_id"]
budget = lead.iloc[0]["budget"]
engagement = lead.iloc[0]["engagement_score"]

print(f"Lead selected: {lead_id}")
print(f"Budget: {budget:.2f}")
print(f"Engagement: {engagement}")

# --------------------------------------------------
# تحميل العقارات
# --------------------------------------------------

properties_query = """
SELECT
property_id,
price,
market_estimated_price,
bedrooms,
sqft,
location_lat,
location_long
FROM properties
"""

properties = pd.read_sql(properties_query,engine)

print("Properties loaded:",len(properties))

# --------------------------------------------------
# Popularity Score
# --------------------------------------------------

popularity_query = """
SELECT
property_id,
SUM(CASE WHEN interaction_type='view' THEN 1 ELSE 0 END) views,
SUM(CASE WHEN interaction_type='favorite' THEN 1 ELSE 0 END) favorites,
SUM(CASE WHEN interaction_type='inquiry' THEN 1 ELSE 0 END) inquiries
FROM property_interactions
GROUP BY property_id
"""

popularity = pd.read_sql(popularity_query,engine)

if popularity.empty:

    properties["popularity_score"] = 0

else:

    popularity["popularity_score"] = (
        popularity["views"]*1 +
        popularity["favorites"]*3 +
        popularity["inquiries"]*5
    )

    properties = properties.merge(
        popularity[["property_id","popularity_score"]],
        on="property_id",
        how="left"
    )

    properties["popularity_score"] = properties["popularity_score"].fillna(0)

# --------------------------------------------------
# Deal Detection
# --------------------------------------------------

properties["price_per_sqft"] = properties["price"]/properties["sqft"]

avg_sqft_price = properties["price_per_sqft"].mean()

properties["sqft_value"] = (
    avg_sqft_price - properties["price_per_sqft"]
)/avg_sqft_price

properties["price_gap"] = (
    properties["market_estimated_price"] - properties["price"]
)

properties["price_gap_ratio"] = (
    properties["price_gap"] /
    properties["market_estimated_price"]
)

properties["deal_score"] = (
    properties["price_gap_ratio"]*0.6 +
    properties["sqft_value"]*0.4
)

def classify_deal(v):

    if v > 0.18:
        return "🔥 Best Deal"

    elif v > 0.08:
        return "💰 Good Deal"

    elif v > -0.05:
        return "⚖️ Fair Price"

    else:
        return "⚠️ Overpriced"

properties["deal_label"] = properties["deal_score"].apply(classify_deal)

# --------------------------------------------------
# Budget Matching
# --------------------------------------------------

properties["budget_gap"] = abs(properties["price"] - budget)

# --------------------------------------------------
# Location Distance
# --------------------------------------------------

lead_lat = 36.7783
lead_lon = -119.4179

def haversine(lat1,lon1,lat2,lon2):

    R = 6371

    lat1=np.radians(lat1)
    lon1=np.radians(lon1)
    lat2=np.radians(lat2)
    lon2=np.radians(lon2)

    dlat=lat2-lat1
    dlon=lon2-lon1

    a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c=2*np.arcsin(np.sqrt(a))

    return R*c

properties["distance"]=haversine(
    lead_lat,
    lead_lon,
    properties["location_lat"],
    properties["location_long"]
)

# --------------------------------------------------
# اختيار الميزات
# --------------------------------------------------

features=[
"budget_gap",
"bedrooms",
"sqft",
"price_gap",
"price_per_sqft",
"distance",
"popularity_score"
]

properties_features=properties[features]

lead_features=pd.DataFrame([[
0,
3,
1500,
0,
budget/1500,
0,
properties["popularity_score"].mean()
]],columns=features)

# --------------------------------------------------
# Normalize
# --------------------------------------------------

scaler=MinMaxScaler()

combined=pd.concat([properties_features,lead_features])

scaled=scaler.fit_transform(combined)

properties_scaled=scaled[:-1]
lead_scaled=scaled[-1].reshape(1,-1)

# --------------------------------------------------
# Feature Weights
# --------------------------------------------------

weights=np.array([

3.0,  # budget match
1.5,  # bedrooms
1.0,  # sqft
2.0,  # price gap
1.2,  # price per sqft
1.8,  # distance
1.2   # popularity

])

properties_weighted=properties_scaled*weights
lead_weighted=lead_scaled*weights

# --------------------------------------------------
# Similarity
# --------------------------------------------------

similarity=cosine_similarity(
lead_weighted,
properties_weighted
)

properties["score"]=similarity[0]

# --------------------------------------------------
# Collaborative Boost
# --------------------------------------------------

collab_query=f"""
SELECT DISTINCT property_id
FROM property_interactions
WHERE lead_id IN(

SELECT lead_id
FROM leads
WHERE ABS(budget-{budget})<20000

)
AND interaction_type='favorite'
"""

collab=pd.read_sql(collab_query,engine)

if not collab.empty:

    properties.loc[
        properties["property_id"].isin(collab["property_id"]),
        "score"
    ]+=0.05

# --------------------------------------------------
# تحديد عدد النتائج
# --------------------------------------------------

if engagement>0.8:

    top_n=3

elif engagement>0.5:

    top_n=5

else:

    top_n=8

recommendations=properties.sort_values(
by="score",
ascending=False
).head(top_n)

# --------------------------------------------------
# عرض النتائج
# --------------------------------------------------

print("\nTop Recommended Properties\n")

print(recommendations[[

"property_id",
"price",
"bedrooms",
"sqft",
"distance",
"price_gap",
"deal_label",
"score"

]])

# --------------------------------------------------
# حفظ النتائج
# --------------------------------------------------

recommendations["lead_id"]=lead_id
recommendations["created_at"]=datetime.now()

recommendations[[

"lead_id",
"property_id",
"score",
"created_at"

]].to_sql(
"lead_recommendations",
engine,
if_exists="append",
index=False
)



deal_model = joblib.load("deal_model.pkl")

predicted_prices = deal_model.predict(
properties[["sqft","bedrooms"]]
)

properties["predicted_market_price"] = predicted_prices

properties["deal_value"] = (
properties["predicted_market_price"] -
properties["price"]
)

print("\nRecommendations saved successfully!")