import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" Enhanced XGBoost Employee Matcher Training")
print("="*70)

def load_training_data():
    """Load training data from CSV"""
    try:
        df = pd.read_csv('enhanced_training_data.csv')
        print(f"\n✓ Loaded {len(df)} training samples from enhanced_training_data.csv")
        return df
    except FileNotFoundError:
        print("✗ Enhanced data file not found. Please run generate_enhanced_training_data.py first")
        return None

def prepare_features(df):
    """Prepare features for training"""
    
    if 'deal_success' in df.columns and 'target' not in df.columns:
        df['target'] = df['deal_success']
    
    feature_cols = [

        'budget_match',
        'location_match', 
        'property_type_match',
        'productivity',
        'average_response_time',
        'win_rate',
        'loss_rate',
        'deals_won',
        'deals_lost',
        'total_deals_closed',
        'response_time_score',
        
        'price',
        'sqft',
        'bedrooms',
        'bathrooms',
        'days_on_market',
        'neighborhood',
        'property_appeal',
        
        'total_deals_as_client',
        'preferred_budget_min',
        'preferred_budget_max',
        

        'price_fit'
    ]
    

    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  Missing columns: {missing_cols}")
        print(f" Using available columns: {len(available_cols)}")
    
    if 'target' not in df.columns:
        print("✗ No target column found!")
        return None, None, None, None
    
    X = df[available_cols]
    y = df['target']
    
    print(f"✓ Features: {len(available_cols)} columns")
    print(f"✓ Target: {(y == 1).sum()} positive, {(y == 0).sum()} negative")
    
    return X, y, available_cols, df

def train_model():
    """Train the improved XGBoost model"""
    
    df = load_training_data()
    if df is None or df.empty:
        return
    
    print("\n" + "="*70)
    print(" Data Analysis")
    print("="*70)
    
    if 'deal_success' in df.columns:
        target_col = 'deal_success'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        print("✗ No target column found!")
        return
    
    print(f"Total samples: {len(df)}")
    print(f"Win rate: {(df[target_col] == 1).sum() / len(df) * 100:.1f}%")
    
    if 'employee_id' in df.columns:
        print(f"Employees: {df['employee_id'].nunique()}")
    
    if 'property_type' in df.columns:
        print(f"Property types: {df['property_type'].nunique()}")
    
    if 'client_preferred_location' in df.columns:
        print(f"Locations: {df['client_preferred_location'].nunique()}")
    
    print("\n" + "="*70)
    print(" Feature Engineering")
    print("="*70)
    
    X, y, feature_cols, df_full = prepare_features(df)
    
    if X is None:
        return
    
    print("\n" + "="*70)
    print(" Model Training")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train positive rate: {y_train.mean():.1%}")
    print(f"Test positive rate: {y_test.mean():.1%}")
    
    print("\nHandling missing values...")
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    print("✓ Missing values filled with mean")
    
    print("\nTraining XGBoost with optimized parameters...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.5,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    print("✓ Model training complete")
    
    print("\n" + "="*70)
    print(" Model Evaluation")
    print("="*70)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nTest Metrics:")
    print(f"  AUC-ROC:  {auc:.4f} {'⬆️' if auc > 0.67 else '⬇️'} (target: 0.75+)")
    print(f"  Accuracy: {acc:.4f} {'⬆️' if acc > 0.40 else '⬇️'} (target: 0.75+)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\n" + "="*70)
    print(" Feature Importance")
    print("="*70)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    for idx, row in feature_importance.head(10).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = "█" * bar_length
        print(f"  {row['feature']:30s} : {bar} {row['importance']:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/employee_matcher_xgb.pkl")
    joblib.dump(feature_cols, "models/employee_matcher_features.pkl")
    joblib.dump(feature_importance, "models/feature_importance.pkl")
    
    print("\n" + "="*70)
    print(" Model Artifacts Saved")
    print("="*70)
    print("✓ models/employee_matcher_xgb.pkl")
    print("✓ models/employee_matcher_features.pkl")
    print("✓ models/feature_importance.pkl")
    
    print("\n" + "="*70)
    print(" Summary")
    print("="*70)
    print(f"\nModel Performance: {auc:.2%} AUC")
    print(f"Improvement: {((auc - 0.6667) / 0.6667 * 100):.1f}% from baseline")
    print(f"\nReady to use for recommendations!")

if __name__ == "__main__":
    train_model()
