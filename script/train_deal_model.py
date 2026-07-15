import pandas as pd
import numpy as np
import joblib
import os
import logging
from sqlalchemy import create_engine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = "mysql+pymysql://root:@127.0.0.1/real_estate_ai"

def load_data():
    query = """
        SELECT budget, urgency_level, source, engagement_score, price_gap, conversion_status
        FROM leads
        WHERE conversion_status IS NOT NULL
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} leads")
    return df

def prepare_features(df):
    df = pd.get_dummies(df, columns=["urgency_level", "source"])
    X = df.drop("conversion_status", axis=1)
    y = df["conversion_status"]
    feature_names = list(X.columns.astype(str))
    return X, y, feature_names

def train_and_save():
    df = load_data()
    if df.empty:
        logger.error("No data found. Exiting.")
        return

    X, y, feature_names = prepare_features(df)
    logger.info(f"Features shape: {X.shape}, target distribution: {y.value_counts().to_dict()}")

   
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        accuracies.append(acc)
        logger.info(f"\nFold {fold} Accuracy: {acc:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))

    logger.info(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")

    
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lead_model.pkl")
    joblib.dump(feature_names, "models/lead_model_columns.pkl")
    logger.info("Model saved to models/lead_model.pkl")

if __name__ == "__main__":
    train_and_save()