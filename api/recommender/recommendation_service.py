import pandas as pd
from .user_stage import UserStageDetector
from .hybrid_recommender import HybridRecommender


class RecommendationService:
    def __init__(self, engine=None):
        self.stage_detector = UserStageDetector(engine=engine)
        self.hybrid = HybridRecommender(self.stage_detector.engine)

    def recommend(self, user_id, top_n: int = 10, user_type=None):
        stage = self.stage_detector.get_stage(user_id, user_type=user_type)
        sections = self.hybrid.get_recommendations(user_id, stage, top_n=top_n)

        trending = sections.get("trending") or []
        for_you = sections.get("for_you") or []
        similar = sections.get("similar") or []

        return {
            "user_id": str(user_id),
            "user_stage": stage,
            "strategy": self._strategy_name(stage),
            "sections": {
                "trending": trending,
                "for_you": for_you,
                "similar": similar,
            },
            "counts": {
                "trending": len(trending),
                "for_you": len(for_you),
                "similar": len(similar),
                "total": len(trending) + len(for_you) + len(similar),
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def _strategy_name(self, stage: str) -> str:
        stage = (stage or "").upper()
        if stage == "VISITOR":
            return "trending_only"
        if stage in ("LEAD", "WARM_LEAD", "HOT_LEAD"):
            return "content_plus_trending"
        if stage == "CLIENT":
            return "hybrid_content_collaborative_trending"
        return "trending_only"
