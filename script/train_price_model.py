import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(" Training OPTIMIZED Price Model on Ames Housing...")

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

query = """
SELECT price, overall_qual, overall_cond, year_built, year_remodel, sqft,
       bsmt_sf, fireplaces, garage_cars, garage_area, deck_sf, porch_sf,
       ms_zoning, neighborhood
FROM properties
WHERE price > 50000 AND sqft > 0
"""

data = pd.read_sql(query, engine)

data['log_price'] = np.log1p(data['price'])

features = ["overall_qual", "overall_cond", "year_built", "year_remodel", "sqft",
            "bsmt_sf", "fireplaces", "garage_cars", "garage_area", "deck_sf",
            "porch_sf", "ms_zoning", "neighborhood"]

X = data[features]
y = data['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=8,
                         subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print(f"R² Score : {r2:.4f}")
print(f"MAE      : ${mae:,.0f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/price_model.pkl")
joblib.dump(features, "models/price_features.pkl")

print(" Price model SAVED (High Performance)")