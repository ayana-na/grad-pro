import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from .feature_engineering import prepare_conversion_features
import os

class AIServices:
    def __init__(self, config):
        
        self.config = config
        self.engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
        
        self.conversion_model = joblib.load(config.CONVERSION_MODEL_PATH)
        self.conversion_features = joblib.load(config.CONVERSION_FEATURES_PATH)
        self.employee_model = None
        self.employee_model = joblib.load(config.EMPLOYEE_MODEL_PATH)
        self.employee_features = joblib.load(config.EMPLOYEE_FEATURES_PATH)
        self.employee_encoders = joblib.load(config.EMPLOYEE_ENCODERS_PATH)

    def predict_conversion(self, lead_id):
        query = """
        SELECT 
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
        WHERE l.lead_id = %s
        """
        df = pd.read_sql(query, self.engine, params=(lead_id,))
        if df.empty:
            return None
        
        df_feat = prepare_conversion_features(df)
        df_encoded = pd.get_dummies(df_feat, columns=["urgency_level", "source", "neighborhood"], drop_first=True)

        for col in self.conversion_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        X = df_encoded[self.conversion_features]
        prob = self.conversion_model.predict_proba(X)[:, 1][0]
        return float(prob)
    
    def segment_lead(self, lead_id):
        query = "SELECT segment FROM customer_segments WHERE lead_id = %s"
        df = pd.read_sql(query, self.engine, params=(lead_id,))
        if df.empty:
            return None
        return df.iloc[0]['segment']
    
    def recommend_employees_for_deal(self, deal_id, top_n=5):
        if self.employee_model is None:
            return []
        
        
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
        deal_df = pd.read_sql(query, self.engine, params=(deal_id,))
        if deal_df.empty:
            return []
        deal = deal_df.iloc[0]
        
        
        employees = pd.read_sql("SELECT * FROM employees WHERE type = 'SALES'", self.engine)
        if employees.empty:
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
                if col_enc in self.employee_features:
                    encoder = self.employee_encoders.get(col)
                    if encoder:
                        if col == 'preferred_location':
                            val = str(deal['preferred_location'])
                        elif col == 'neighborhood':
                            val = str(deal['neighborhood'])
                        elif col == 'employee_preferred_location':
                            val = str(emp['preferred_location'])
                        elif col == 'property_type':
                            val = str(deal['property_type'])
                        elif col == 'employee_type':
                            val = str(emp['type'])
                        else:
                            val = ''
                        if val in encoder.classes_:
                            row[col_enc] = encoder.transform([val])[0]
                        else:
                            row[col_enc] = 0
                    else:
                        row[col_enc] = 0
                else:
                    row[col_enc] = 0
            
            try:
                X_row = pd.DataFrame([row])[self.employee_features]
                prob = self.employee_model.predict_proba(X_row)[0, 1]
                scores.append((emp['employee_id'], prob))
            except Exception as e:
                print(f"Error predicting for employee {emp['employee_id']}: {e}")
                continue
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{'employee_id': int(eid), 'score': float(score)} for eid, score in scores[:top_n]]