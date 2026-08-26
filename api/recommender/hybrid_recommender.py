from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender


class HybridRecommender:
    def __init__(self, engine):
        self.engine = engine
        self.content = ContentBasedRecommender(engine)
        self.collab = CollaborativeRecommender(engine)

    def get_recommendations(self, user_id, stage: str, top_n: int = 10):
    
        stage = (stage or "VISITOR").upper()
        user_id = str(user_id)

        trending = self._tag_source(self._trending(top_n), "trending")
        for_you = []
        similar = []

        if stage in ("LEAD", "WARM_LEAD", "HOT_LEAD", "CLIENT"):
            for_you = self._tag_source(
                self.content.recommend_for_lead(user_id, top_n=top_n),
                "content",
            )

        if stage == "CLIENT":
          similar = self._tag_source(
            self.collab.recommend_for_user(user_id, top_n=top_n),
             "collaborative",
    )

        for_you = self._exclude_seen(for_you, trending)
        similar = self._exclude_seen(similar, trending + for_you)

        return {
            "trending": trending[:top_n],
            "for_you": for_you[:top_n],
            "similar": similar[:top_n],
        }

    def _tag_source(self, recs, source: str):
        out = []
        for r in recs or []:
            item = dict(r)
            item["source"] = source
            if source == "trending":
                item["reason"] = item.get("reason") or "Currently popular among buyers"
            elif source == "content":
                item["reason"] = item.get("reason") or "Matched to your preferences"
            elif source == "collaborative":
                item["reason"] = item.get("reason") or "Similar users also liked this property"
            out.append(item)
        return out

    def _exclude_seen(self, recs, seen_list):
        seen = {str(r.get("property_id")) for r in (seen_list or [])}
        return [r for r in (recs or []) if str(r.get("property_id")) not in seen]

    def _trending(self, top_n: int = 10):
        import pandas as pd

        query = """
            SELECT
                p.id AS property_id,
                p.listed_price AS price,
                p.sqft,
                p.num_of_rooms AS bedrooms,
                COUNT(lb.id) AS interactions,
                'Currently popular among buyers' AS reason,
                COUNT(lb.id)::float AS score
            FROM properties p
            LEFT JOIN lead_behaviors lb ON lb.property_id = p.id
            WHERE p.status = 'AVAILABLE'
            GROUP BY p.id, p.listed_price, p.sqft, p.num_of_rooms
            ORDER BY interactions DESC NULLS LAST, p.listed_price ASC
            LIMIT %(limit)s
        """
        try:
            df = pd.read_sql(query, self.engine, params={"limit": top_n})
        except Exception as e:
            print(f"Trending query error: {e}")
            return []

        recs = []
        for _, row in df.iterrows():
            recs.append({
                "property_id": str(row["property_id"]),
                "price": float(row["price"]) if row.get("price") is not None else None,
                "sqft": float(row["sqft"]) if row.get("sqft") is not None else None,
                "bedrooms": int(row["bedrooms"]) if row.get("bedrooms") is not None else None,
                "score": float(row.get("score") or 0),
                "reason": row.get("reason") or "Currently popular among buyers",
            })
        return recs
