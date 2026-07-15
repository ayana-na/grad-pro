import pandas as pd
import logging
from sqlalchemy import create_engine
from typing import List

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")
        self._cached_properties = None
        self._cached_lead_interactions = None

    def load_properties(self) -> pd.DataFrame:
        if self._cached_properties is not None:
            return self._cached_properties

        query = """
            SELECT 
                property_id, price, sqft, bedrooms, bathrooms,
                location_lat, location_long, neighborhood, year_built,
                overall_qual, overall_cond, garage_cars, fireplaces,
                deck_sf, bsmt_sf, listing_date
            FROM properties
        """
        df = pd.read_sql(query, self.engine)
        df = df.dropna(subset=['price', 'sqft'])
        df['price'] = df['price'].astype(float)
        df['sqft'] = df['sqft'].astype(float)
        self._cached_properties = df
        logger.info(f"Loaded {len(df)} properties")
        return df

    def load_lead_interactions(self) -> pd.DataFrame:
        if self._cached_lead_interactions is not None:
            return self._cached_lead_interactions

        query = """
            SELECT 
                lead_id,
                property_id,
                conversion_probability,
                engagement_score
            FROM leads
            WHERE conversion_probability IS NOT NULL
        """
        df = pd.read_sql(query, self.engine)
        df['lead_id'] = df['lead_id'].astype(int)
        df['property_id'] = df['property_id'].astype(int)
        df['conversion_probability'] = df['conversion_probability'].astype(float)
        df['engagement_score'] = df['engagement_score'].astype(float)
        self._cached_lead_interactions = df
        logger.info(f"Loaded {len(df)} lead interactions")
        return df

    def get_trending_properties(self, num: int = 10) -> pd.DataFrame:
        query = """
            SELECT 
                p.property_id,
                p.price,
                p.sqft,
                p.bedrooms,
                COUNT(l.lead_id) as lead_count,
                AVG(l.conversion_probability) as avg_prob,
                (COUNT(l.lead_id) * 0.6 + COALESCE(AVG(l.conversion_probability), 0) * 40) as score,
                'Currently the most popular property' as reason
            FROM properties p
            LEFT JOIN leads l ON p.property_id = l.property_id
            GROUP BY p.property_id, p.price, p.sqft, p.bedrooms
            ORDER BY score DESC
            LIMIT %s
        """
        return pd.read_sql(query, self.engine, params=(num,))

    def get_all_leads(self) -> List[int]:
        df = pd.read_sql("SELECT lead_id FROM leads WHERE lead_category IN ('HOT', 'WARM')", self.engine)
        return df['lead_id'].tolist()

    def get_lead_interacted_properties(self, lead_id: int) -> pd.DataFrame:
        query = """
            SELECT property_id 
            FROM leads 
            WHERE lead_id = %s
        """
        return pd.read_sql(query, self.engine, params=(lead_id,))