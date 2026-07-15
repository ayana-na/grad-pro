# evaluate_model.py
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model():
    engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

    model = joblib.load("models/conversion_model_optimized.pkl")
    conversion_features = joblib.load("models/conversion_features_optimized.pkl")

    query = """
    SELECT 
        l.budget AS price,
        p.sqft,
        p.bedrooms,
        p.bathrooms,
        DATEDIFF(CURDATE(), p.listing_date) AS days_on_market,
        p.overall_qual,
        p.overall_cond,
        p.year_built,
        p.neighborhood,
        p.garage_cars,
        p.bsmt_sf,
        p.fireplaces,
        p.deck_sf,
        l.engagement_score,
        l.price_gap,
        l.conversion_status AS target
    FROM leads l
    JOIN properties p ON l.property_id = p.property_id
    WHERE l.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        AND l.conversion_status IS NOT NULL
    """
    data = pd.read_sql(query, engine).dropna()
    if len(data) == 0:
        logger.warning("No recent data for evaluation")
        return

    data['price_per_sqft'] = data['price'] / data['sqft']
    data['property_age'] = 2026 - data['year_built']
    data['quality_index'] = data['overall_qual'] * data['overall_cond']
    data['luxury_score'] = (data['garage_cars'] * 2) + (data['fireplaces'] * 1.5) + (data['deck_sf'] / 100)
    data['price_to_income_ratio'] = data['price'] / (data['engagement_score'] * 100000 + 1)
    data['days_on_market_log'] = np.log1p(data['days_on_market'])
    data['sqft_per_bedroom'] = data['sqft'] / (data['bedrooms'] + 1)
    data['bath_per_bedroom'] = data['bathrooms'] / (data['bedrooms'] + 1)
    data['has_basement'] = (data['bsmt_sf'] > 0).astype(int)
    data['has_fireplace'] = (data['fireplaces'] > 0).astype(int)
    data['has_garage'] = (data['garage_cars'] > 0).astype(int)

    categorical_cols = ['neighborhood']
    cols_to_encode = [col for col in categorical_cols if col in data.columns]
    data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)

    feature_cols = [col for col in conversion_features if col in data.columns]
    X = data[feature_cols]
    y_true = data['target']

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
        'samples': len(y_true),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_sql('model_performance', engine, if_exists='append', index=False)
    logger.info(f"Evaluation metrics saved: {metrics}")

if __name__ == "__main__":
    evaluate_model()