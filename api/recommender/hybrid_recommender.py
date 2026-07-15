from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, engine):
        self.content = ContentBasedRecommender(engine)
        self.collab = CollaborativeRecommender(engine)

    def get_recommendations(self, user_id: int, stage: str, top_n: int = 10):
        
        if stage == 'CLIENT':
            content_recs = self.content.recommend_for_lead(user_id, top_n=20)
            collab_recs = self.collab.recommend_for_lead(user_id, top_n=15)
            return self._merge_recommendations(content_recs, collab_recs)[:top_n]
        
        else:
            return self.content.recommend_for_lead(user_id, top_n=top_n)

    def _merge_recommendations(self, content, collab):
        seen = {rec['property_id']: rec for rec in content}
        
        for rec in collab:
            pid = rec['property_id']
            if pid in seen:
                seen[pid]['score'] = seen[pid].get('score', 0) + rec.get('score', 0) * 1.3
                seen[pid]['reason'] = "Recommended for you (Hybrid)"
            else:
                seen[pid] = rec
                seen[pid]['reason'] = "Similar clients liked this"
                
        return sorted(seen.values(), key=lambda x: x.get('score', 0), reverse=True)