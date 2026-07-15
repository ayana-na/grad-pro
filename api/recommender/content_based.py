import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class ContentBasedRecommender:
    def __init__(self, engine):
        self.engine = engine
        self.properties = None
        self.feature_matrix = None
        self._load_and_build()

    def _load_and_build(self):
       
        query = """
            SELECT property_id, price, sqft, bedrooms, bathrooms, 
                   overall_qual, garage_cars, year_built, neighborhood
            FROM properties
        """
        df = pd.read_sql(query, self.engine)
        self.properties = df.copy().reset_index(drop=True)
        
       
        self.properties['sqft'] = self.properties['sqft'].clip(lower=800, upper=4500)
        self.properties['age'] = 2026 - self.properties['year_built'].fillna(2000)
        self.properties['luxury_score'] = self.properties['garage_cars'].fillna(0) * 2
        
        
        features = self.properties[['price', 'sqft', 'bedrooms', 'bathrooms', 
                                   'overall_qual', 'luxury_score', 'age']]
        
        scaler = MinMaxScaler()
        self.feature_matrix = scaler.fit_transform(features)

    def recommend_for_lead(self, lead_id: int, top_n: int = 12):
       
        lead_info = pd.read_sql("""
            SELECT budget, engagement_score 
            FROM leads WHERE lead_id = %s
        """, self.engine, params=(lead_id,))
        
        budget = float(lead_info['budget'].iloc[0]) if not lead_info.empty else None
        engagement = float(lead_info.get('engagement_score', pd.Series([0.5])).iloc[0])

       
        if budget:
            price_mask = np.abs(self.properties['price'] - budget) / budget <= 0.33
            candidates = self.properties[price_mask].copy()
        else:
            candidates = self.properties.copy()

        if candidates.empty:
            candidates = self.properties.copy()

        
        candidates = candidates.copy()
        candidates['score'] = (candidates['overall_qual'].fillna(5) / 10) * 0.4
        
        if budget:
            budget_factor = 1 - (np.abs(candidates['price'] - budget) / budget) * 0.45
            candidates['score'] += budget_factor * 0.6

       
        candidates['expected_revenue'] = candidates['price'] * (0.022 + engagement * 0.038)
        candidates['priority_score'] = candidates['score'] * (0.75 + engagement * 0.5)

       
        candidates = self._apply_diversity(candidates, top_n)

        
        top_props = candidates.nlargest(top_n, 'priority_score')

        return self._format_recommendations(top_props)

    def _apply_diversity(self, df, limit):
        
        if len(df) <= limit:
            return df
            
        df = df.copy()
        df['price_group'] = pd.qcut(df['price'], q=5, labels=False, duplicates='drop')
        
        diverse = []
        for g in range(5):
            group = df[df['price_group'] == g]
            if not group.empty:
                diverse.append(group.nlargest(4, 'priority_score'))
        
        result = pd.concat(diverse).drop_duplicates('property_id')
        return result.head(limit * 2)

    def _format_recommendations(self, df, top_n=12):
        recs = []
        for _, row in df.head(top_n).iterrows():
            if row['score'] >= 0.88:
                reason = "Excellent match - Strong budget & quality fit"
            elif row['score'] >= 0.78:
                reason = "Very good match for your budget"
            else:
                reason = "Good opportunity in your price range"
            
            recs.append({
                "property_id": int(row['property_id']),
                "price": float(row['price']),
                "sqft": int(row.get('sqft', 0)),
                "bedrooms": int(row.get('bedrooms', 0)),
                "score": round(float(row['score']), 3),
                "expected_revenue": round(float(row['expected_revenue']), 0),
                "priority_score": round(float(row['priority_score']), 3),
                "reason": reason
            })
        return recs