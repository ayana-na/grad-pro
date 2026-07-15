import pandas as pd
from sqlalchemy import create_engine
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

print(" Training FINAL Lead Model (XGBoost - Safe Save)...")

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

query = """
SELECT budget, urgency_level, source, engagement_score, price_gap, conversion_status
FROM leads
WHERE conversion_status IS NOT NULL
"""

df = pd.read_sql(query, engine)

df = pd.get_dummies(df, columns=["urgency_level", "source"])
X = df.drop("conversion_status", axis=1)
y = df["conversion_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='auc'
)

model.fit(X_train, y_train)

print("\n" + "="*50)
print("Lead Model Result:")
print(classification_report(y_test, model.predict(X_test)))
print("="*50)


os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/lead_model.pkl")
joblib.dump(list(X.columns), "models/lead_model_columns.pkl")   

print(" Lead model SAVED successfully!")