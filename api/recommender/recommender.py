import pandas as pd
import logging
import time
from typing import List, Dict
from datetime import datetime

from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from data_loader import DataLoader

logger = logging.getLogger(__name__)

class RealEstateRecommender:
    def __init__(self):
        self.data_loader = DataLoader()
        properties_df = self.data_loader.load_properties()

        self.content_model = ContentBasedRecommender(
            properties_df, self.data_loader.engine
        )
        
        self.collaborative_model = CollaborativeRecommender(
            self.data_loader.load_lead_interactions(),
            properties_df
        )
        
        self.cache = {}
        self.cache_timestamps = {}

    def get_recommendations(self, lead_id: int, num_recommendations: int = 10) -> List[Dict]:
        cache_key = f"lead_{lead_id}"
        if cache_key in self.cache and time.time() - self.cache_timestamps.get(cache_key, 0) < 3600:
            return self.cache[cache_key]

        try:
            content_recs = self.content_model.recommend_for_lead(lead_id, 60)
            collab_recs = self.collaborative_model.recommend_for_lead(lead_id, 30)

            combined = self._merge_recommendations(content_recs, collab_recs)

            if len(combined) < 6:
                trending = self.get_trending_properties(num_recommendations)
                combined.extend(trending[:num_recommendations - len(combined)])

            combined = self._apply_strong_diversity(combined, num_recommendations)

            for rec in combined:
                if isinstance(rec.get('reason'), str) and ' (score:' in rec['reason']:
                    rec['reason'] = rec['reason'].split(' (score:')[0]

            self.cache[cache_key] = combined[:num_recommendations]
            self.cache_timestamps[cache_key] = time.time()

            return combined[:num_recommendations]

        except Exception as e:
            logger.error(f"Error generating recommendations for lead {lead_id}: {e}")
            return self.get_trending_properties(num_recommendations)

    def _merge_recommendations(self, content: List, collab: List) -> List:
        seen = {}

        for rec in content:
            pid = rec['property_id']
            seen[pid] = rec.copy()
            seen[pid]['score'] = rec.get('score', 0) * 0.40

        for rec in collab:
            pid = rec['property_id']
            collab_score = rec.get('score', 0) * 1.95

            if pid in seen:
                seen[pid]['score'] += collab_score
                if collab_score > 0.08:
                    seen[pid]['reason'] = "Recommended by similar users"
            else:
                seen[pid] = rec.copy()
                seen[pid]['score'] = collab_score
                seen[pid]['reason'] = "Recommended by similar users"

        return sorted(seen.values(), key=lambda x: x['score'], reverse=True)

    def _apply_strong_diversity(self, recommendations: List[Dict], max_recs: int = 10) -> List[Dict]:
        if len(recommendations) <= 5:
            return recommendations[:max_recs]

        final = []
        price_groups = {}

        for rec in recommendations:
            if len(final) >= max_recs:
                break

            p_group = round(rec.get('price', 0) / 10000) * 10000

            if price_groups.get(p_group, 0) < 1:
                final.append(rec)
                price_groups[p_group] = price_groups.get(p_group, 0) + 1

        if len(final) < max_recs:
            for rec in recommendations:
                if rec not in final:
                    final.append(rec)
                    if len(final) >= max_recs:
                        break

        return final

    def get_trending_properties(self, top_n: int = 10) -> List[Dict]:
        df = self.data_loader.get_trending_properties(top_n)
        records = df.to_dict(orient='records')
        for rec in records:
            rec['score'] = rec.get('score', 0) / 25.0
            rec['reason'] = "Currently popular among buyers"
        return records

    def update_all_recommendations(self):
        leads = self.data_loader.get_all_leads()
        for lead_id in leads:
            self.get_recommendations(lead_id)
        logger.info(f"Updated recommendations for {len(leads)} leads")

    def explain_recommendation(self, lead_id: int, property_id: int) -> Dict:
        return {
            "lead_id": lead_id,
            "property_id": property_id,
            "reasons": [
                {"type": "content", "description": "Similar to properties you interacted with"},
                {"type": "collaborative", "description": "Users similar to you liked this property"}
            ],
            "generated_at": datetime.now().isoformat()
        }