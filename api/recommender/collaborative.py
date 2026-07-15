import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class CollaborativeRecommender:
    def __init__(self, engine):
        self.engine = engine
        self.svd_model = None
        self.user_item_matrix = None
        self.users = None
        self.items = None
        self._build_model()

    def _build_model(self):
        try:
            query = """
                SELECT lead_id, property_id, conversion_probability 
                FROM leads 
                WHERE conversion_probability IS NOT NULL
            """
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                logger.warning("No interactions found for collaborative model")
                return

            self.user_item_matrix = df.pivot_table(
                index='lead_id',
                columns='property_id',
                values='conversion_probability',
                fill_value=0
            )

            self.users = self.user_item_matrix.index.tolist()
            self.items = self.user_item_matrix.columns.tolist()

            n_components = min(50, len(self.items) - 1, len(self.users) - 1)
            if n_components < 1:
                return
                
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd_model.fit(self.user_item_matrix)
            
            logger.info(f" Collaborative model trained successfully with {n_components} components")
            
        except Exception as e:
            logger.error(f" Failed to build collaborative model: {e}")
            self.svd_model = None

    def recommend_for_lead(self, lead_id: int, top_n: int = 10):
        if self.svd_model is None or lead_id not in self.users:
            return [] 

        try:
            user_idx = self.users.index(lead_id)
            user_vector = self.svd_model.transform(self.user_item_matrix.iloc[[user_idx]])
            predicted_scores = np.dot(user_vector, self.svd_model.components_).flatten()


            sorted_indices = np.argsort(predicted_scores)[::-1]
            recommendations = []
            seen = set()

            for idx in sorted_indices:
                if len(recommendations) >= top_n:
                    break
                prop_id = self.items[idx]
                if prop_id in seen:
                    continue
                
                score = predicted_scores[idx]
                if score > 0.1: 
                    recommendations.append({
                        "property_id": int(prop_id),
                        "score": float(score),
                        "reason": "Similar users also liked this property"
                    })
                    seen.add(prop_id)

            return recommendations

        except Exception as e:
            logger.error(f"Collaborative recommendation error: {e}")
            return []