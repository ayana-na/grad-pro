import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import catboost as cb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import matplotlib.pyplot as plt
import warnings
import optuna
from optuna.samplers import TPESampler
warnings.filterwarnings('ignore')

print("="*70)
print(" Optimized Conversion Model with Hyperparameter Tuning")
print("="*70)


USE_TUNING = True   
N_TRIALS = 20       

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

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
WHERE l.conversion_status IS NOT NULL
"""

data = pd.read_sql(query, engine).dropna()
print(f" Samples: {len(data)}")
print(f" Target distribution:\n{data['target'].value_counts()}")


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

feature_cols = [col for col in data.columns if col not in ['target']]
X = data[feature_cols]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': 0
    }
    model = cb.CatBoostClassifier(**params)
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    return scores.mean()


if USE_TUNING:
    print("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print("Best trial:")
    best_params = study.best_params
    print(f"  AUC: {study.best_value:.4f}")
    print(f"  Params: {best_params}")
else:
    best_params = {
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'border_count': 128
    }
    print("Using default parameters (tuning disabled)")


final_model = cb.CatBoostClassifier(**best_params, random_seed=42, verbose=100)
final_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', final_model)
])
final_pipeline.fit(X_train, y_train)


y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]
print("\n" + "="*70)
print("Results on test set:")
print(classification_report(y_test, y_pred))
test_auc = roc_auc_score(y_test, y_proba)
print(f" ROC-AUC: {test_auc:.4f}")


os.makedirs("models", exist_ok=True)
joblib.dump(final_pipeline, "models/conversion_model_optimized.pkl")
joblib.dump(feature_cols, "models/conversion_features_optimized.pkl")
print(" Model saved to models/conversion_model_optimized.pkl")


fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'CatBoost (AUC = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Conversion Model')
plt.legend()
plt.savefig('conversion_roc.png')
plt.show()
print(" ROC curve saved as conversion_roc.png")

print("\n Done!")