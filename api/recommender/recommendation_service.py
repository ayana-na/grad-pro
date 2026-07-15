import pandas as pd
from .user_stage import UserStageDetector
from .hybrid_recommender import HybridRecommender

class RecommendationService:
    def __init__(self):
        self.stage_detector = UserStageDetector()
        self.hybrid = HybridRecommender(self.stage_detector.engine)

    def recommend(self, user_id: int, top_n: int = 10):
        stage = self.stage_detector.get_stage(user_id)
        
        recommendations = self.hybrid.get_recommendations(user_id, stage, top_n)
        
        return {
            "user_id": user_id,
            "user_stage": stage,
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": pd.Timestamp.now().isoformat()
        }