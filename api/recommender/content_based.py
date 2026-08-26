import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ContentBasedRecommender:
    def __init__(self, engine):
        self.engine = engine
        self.properties = None
        self.feature_matrix = None
        self._load_and_build()

    def _load_and_build(self):
        query = """
            SELECT
                id AS property_id,
                listed_price AS price,
                sqft,
                num_of_rooms AS bedrooms,
                bathrooms,
                overall_qual,
                garage_cars,
                construction_year AS year_built,
                neighborhood_score AS neighborhood,
                status
            FROM properties
            WHERE listed_price IS NOT NULL
              AND sqft IS NOT NULL
        """
        df = pd.read_sql(query, self.engine)
        self.properties = df.copy().reset_index(drop=True)

        if self.properties.empty:
            self.feature_matrix = None
            return

        self.properties["sqft"] = self.properties["sqft"].clip(lower=200, upper=20000)
        self.properties["age"] = 2026 - self.properties["year_built"].fillna(2000)
        self.properties["luxury_score"] = self.properties["garage_cars"].fillna(0) * 2

        features = self.properties[
            ["price", "sqft", "bedrooms", "bathrooms", "overall_qual", "luxury_score", "age"]
        ].fillna(0)

        scaler = MinMaxScaler()
        self.feature_matrix = scaler.fit_transform(features)

    def recommend_for_lead(self, lead_id, top_n: int = 12):
        if self.properties is None or self.properties.empty:
            return []

        lead_info = pd.read_sql(
            """
            SELECT
                l.budget,
                pl.engagement_score
            FROM leads l
            LEFT JOIN property_leads pl ON pl.lead_id = l.id
            WHERE l.id::text = %(lead_id)s
            ORDER BY pl.updated_at DESC NULLS LAST
            LIMIT 1
            """,
            self.engine,
            params={"lead_id": str(lead_id)},
        )

        budget = None
        engagement = 0.5
        if not lead_info.empty:
            if pd.notna(lead_info["budget"].iloc[0]):
                budget = float(lead_info["budget"].iloc[0])
            if pd.notna(lead_info["engagement_score"].iloc[0]):
                engagement = float(lead_info["engagement_score"].iloc[0])

        candidates = self.properties.copy()
        if "status" in candidates.columns:
            available = candidates[candidates["status"] == "AVAILABLE"]
            if not available.empty:
                candidates = available

        if budget and budget > 0:
            price_mask = np.abs(candidates["price"] - budget) / budget <= 0.40
            filtered = candidates[price_mask].copy()
            if not filtered.empty:
                candidates = filtered

        candidates = candidates.copy()
        candidates["score"] = (candidates["overall_qual"].fillna(3) / 5.0) * 0.4

        if budget and budget > 0:
            budget_factor = 1 - (np.abs(candidates["price"] - budget) / budget) * 0.45
            candidates["score"] = candidates["score"] + budget_factor.clip(0, 1) * 0.6

        candidates["expected_revenue"] = candidates["price"] * (0.022 + engagement * 0.038)
        candidates["priority_score"] = candidates["score"] * (0.75 + engagement * 0.5)

        candidates = self._apply_diversity(candidates, top_n)
        top_props = candidates.nlargest(min(top_n * 2, len(candidates)), "priority_score")
        return self._format_recommendations(top_props, top_n)

    def _apply_diversity(self, df, limit):
        if len(df) <= limit:
            return df
        df = df.copy()
        try:
            df["price_group"] = pd.qcut(df["price"], q=5, labels=False, duplicates="drop")
        except Exception:
            return df

        parts = []
        for g in df["price_group"].dropna().unique():
            group = df[df["price_group"] == g]
            if not group.empty:
                parts.append(group.nlargest(4, "priority_score"))
        if not parts:
            return df
        result = pd.concat(parts).drop_duplicates("property_id")
        return result

    def _format_recommendations(self, df, top_n=12):
        recs = []
        for _, row in df.head(top_n).iterrows():
            score = float(row.get("score", 0) or 0)
            if score >= 0.88:
                reason = "Excellent match - Strong budget & quality fit"
            elif score >= 0.78:
                reason = "Very good match for your budget"
            else:
                reason = "Good opportunity in your price range"

            recs.append({
                "property_id": str(row["property_id"]),
                "price": float(row["price"]) if pd.notna(row["price"]) else None,
                "sqft": float(row["sqft"]) if pd.notna(row.get("sqft")) else None,
                "bedrooms": int(row["bedrooms"]) if pd.notna(row.get("bedrooms")) else None,
                "score": round(score, 3),
                "expected_revenue": round(float(row.get("expected_revenue", 0) or 0), 0),
                "priority_score": round(float(row.get("priority_score", 0) or 0), 3),
                "reason": reason,
            })
        return recs
