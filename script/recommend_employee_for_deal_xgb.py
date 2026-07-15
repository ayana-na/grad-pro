import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import joblib
import os
import sys

print("="*60)
print(" Employee Recommender for Deal (XGBoost)")
print("="*60)

engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

def load_model():
    model_path = "models/employee_matcher_xgb.pkl"
    features_path = "models/employee_matcher_features.pkl"
    encoders_path = "models/employee_matcher_encoders.pkl"
    
    if not os.path.exists(model_path):
        print("Model not found. Training now...")
        from train_employee_matcher_xgb import train_model
        train_model()
    
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    encoders = joblib.load(encoders_path)
    return model, feature_cols, encoders

def recommend_employees_for_deal(deal_id, top_n=5):
    
    model, feature_cols, encoders = load_model()
   
    query = """
    SELECT 
        d.id AS deal_id,
        d.client_id,
        d.property_id,
        c.preferred_budget_min,
        c.preferred_budget_max,
        c.preferred_location,
        c.preferred_property_type,
        COALESCE(c.total_deals_as_client, 0) AS total_deals_as_client,
        p.price,
        p.sqft,
        p.bedrooms,
        p.bathrooms,
        p.type AS property_type,
        p.neighborhood,
        DATEDIFF(CURDATE(), p.listing_date) AS days_on_market
    FROM deals d
    JOIN clients c ON d.client_id = c.id
    JOIN properties p ON d.property_id = p.property_id
    WHERE d.id = %s
    """
    deal_df = pd.read_sql(query, engine, params=(deal_id,))
    if deal_df.empty:
        print(f"Deal {deal_id} not found or missing client/property data.")
        return []
    
    deal = deal_df.iloc[0]
    
    employees = pd.read_sql("SELECT * FROM employees WHERE type = 'SALES'", engine)
    if employees.empty:
        print("No sales employees found.")
        return []
    
    scores = []
    for _, emp in employees.iterrows():
        row = {}
        row['budget_range_match'] = 1 if (deal['preferred_budget_min'] is not None and 
                                           deal['preferred_budget_max'] is not None and
                                           deal['price'] >= deal['preferred_budget_min'] and 
                                           deal['price'] <= deal['preferred_budget_max']) else 0
        row['location_match'] = 1 if (deal['preferred_location'] and emp['preferred_location'] and 
                                       deal['preferred_location'] == emp['preferred_location']) else 0
        row['property_type_match'] = 1 if (deal['preferred_property_type'] and deal['property_type'] and 
                                           deal['preferred_property_type'] == deal['property_type']) else 0
        row['total_deals_as_client'] = deal['total_deals_as_client']
        row['price'] = deal['price']
        row['sqft'] = deal['sqft']
        row['bedrooms'] = deal['bedrooms']
        row['bathrooms'] = deal['bathrooms']
        row['days_on_market'] = deal['days_on_market']
        row['productivity'] = emp['productivity']
        row['average_response_time'] = emp['average_response_time']
        row['win_rate'] = emp['deals_won'] / (emp['total_deals_closed'] + 1)
        row['deals_won'] = emp['deals_won']
        row['deals_lost'] = emp['deals_lost']
        
        categorical_cols = ['preferred_location', 'neighborhood', 'employee_preferred_location', 'property_type', 'employee_type']
        for col in categorical_cols:
            col_enc = col + '_enc'
            if col_enc in feature_cols:
                if col == 'preferred_location':
                    val = deal['preferred_location'] or ''
                elif col == 'neighborhood':
                    val = str(deal['neighborhood']) if deal['neighborhood'] is not None else ''
                elif col == 'employee_preferred_location':
                    val = emp['preferred_location'] or ''
                elif col == 'property_type':
                    val = deal['property_type'] or ''
                elif col == 'employee_type':
                    val = emp['type'] or ''
                else:
                    val = ''
                encoder = encoders.get(col)
                if encoder:
                    if val in encoder.classes_:
                        row[col_enc] = encoder.transform([val])[0]
                    else:
                        row[col_enc] = 0
                else:
                    row[col_enc] = 0
        
        X_row = pd.DataFrame([row])[feature_cols]
        prob = model.predict_proba(X_row)[0, 1]  
        scores.append((emp['employee_id'], prob))
    
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores[:top_n]
    
    for rank, (emp_id, score) in enumerate(top_scores, 1):
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO employee_deal_recommendations (deal_id, employee_id, score, rank)
                VALUES (:deal_id, :emp_id, :score, :rank)
                ON DUPLICATE KEY UPDATE score = VALUES(score), rank = VALUES(rank)
            """), {"deal_id": deal_id, "emp_id": emp_id, "score": float(score), "rank": rank})
    
    print(f"\nTop {top_n} recommended employees for Deal ID {deal_id}:")
    for rank, (emp_id, score) in enumerate(top_scores, 1):
        print(f"  {rank}. Employee ID {emp_id} - Success Probability: {score:.4f}")
    
    return top_scores

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recommend_employee_for_deal_xgb.py <deal_id>")
        sys.exit(1)
    try:
        deal_id = int(sys.argv[1])
        recs = recommend_employees_for_deal(deal_id)
        if not recs:
            print("No recommendations generated.")
    except Exception as e:
        print(f"Error: {e}")