import logging
import pandas as pd
import numpy as np
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

    def _load_interactions(self) -> pd.DataFrame:
       
        query = """
        WITH client_direct AS (
            SELECT
                cb.client_id::text AS user_id,
                cb.property_id::text AS property_id,
                CASE cb.event_type::text
                    WHEN 'REQUEST' THEN 1.00
                    WHEN 'SCHEDULE_VISIT' THEN 0.95
                    WHEN 'FAVORITE' THEN 0.90
                    WHEN 'LIKE' THEN 0.80
                    WHEN 'SHARE' THEN 0.75
                    WHEN 'COMPARE' THEN 0.55
                    WHEN 'VIEW' THEN 0.30
                    WHEN 'HIDE' THEN 0.05
                    ELSE 0.25
                END
                + LEAST(COALESCE(cb.dwell_time, 0) / 120.0, 0.25) AS score
            FROM client_behaviors cb
            WHERE cb.property_id IS NOT NULL
              AND cb.client_id IS NOT NULL
              AND cb.event_type::text <> 'HIDE'
        ),
        client_from_lead AS (
            SELECT
                c.id::text AS user_id,
                lb.property_id::text AS property_id,
                CASE lb.event_type::text
                    WHEN 'LIKE' THEN 0.80
                    WHEN 'SHARE' THEN 0.75
                    WHEN 'VIEW' THEN 0.30
                    WHEN 'Notification' THEN 0.20
                    ELSE 0.25
                END
                + LEAST(COALESCE(lb.dwell_time, 0) / 120.0, 0.25) AS score
            FROM clients c
            INNER JOIN lead_behaviors lb ON lb.lead_id = c.lead_id
            WHERE c.lead_id IS NOT NULL
              AND lb.property_id IS NOT NULL
        ),
        client_requests AS (
            SELECT
                r.client_id::text AS user_id,
                r.property_id::text AS property_id,
                CASE UPPER(COALESCE(r.status::text, ''))
                    WHEN 'IN_PROGRESS' THEN 1.00
                    WHEN 'PENDING' THEN 0.85
                    ELSE 0.65
                END AS score
            FROM requests r
            WHERE r.client_id IS NOT NULL
              AND r.property_id IS NOT NULL
        ),
        lead_signals AS (
            SELECT
                pl.lead_id::text AS user_id,
                pl.property_id::text AS property_id,
                GREATEST(
                    COALESCE(pl.conversion_probability, 0.0),
                    COALESCE(pl.engagement_score, 0.0) * 0.5,
                    0.15
                ) AS score
            FROM property_leads pl
            WHERE pl.property_id IS NOT NULL
              AND pl.lead_id IS NOT NULL
        )
        SELECT user_id, property_id, MAX(score) AS score
        FROM (
            SELECT * FROM client_direct
            UNION ALL
            SELECT * FROM client_from_lead
            UNION ALL
            SELECT * FROM client_requests
            UNION ALL
            SELECT * FROM lead_signals
        ) u
        GROUP BY user_id, property_id
        """
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            logger.warning(f"Full collaborative query failed, fallback: {e}")
            return self._load_interactions_fallback()

    def _load_interactions_fallback(self) -> pd.DataFrame:
        query = """
        SELECT
            lead_id::text AS user_id,
            property_id::text AS property_id,
            GREATEST(
                COALESCE(conversion_probability, 0.0),
                COALESCE(engagement_score, 0.0) * 0.5,
                0.15
            ) AS score
        FROM property_leads
        WHERE property_id IS NOT NULL
          AND lead_id IS NOT NULL
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Fallback interactions failed: {e}")
            return pd.DataFrame(columns=["user_id", "property_id", "score"])

    def _build_model(self):
        try:
            df = self._load_interactions()
            if df is None or df.empty:
                logger.warning("No interactions found for collaborative model")
                return

            df["user_id"] = df["user_id"].astype(str)
            df["property_id"] = df["property_id"].astype(str)
            df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

            self.user_item_matrix = df.pivot_table(
                index="user_id",
                columns="property_id",
                values="score",
                aggfunc="max",
                fill_value=0.0,
            )

            self.users = self.user_item_matrix.index.tolist()
            self.items = self.user_item_matrix.columns.tolist()

            if len(self.users) < 2 or len(self.items) < 2:
                logger.warning("Not enough users/items for SVD")
                self.svd_model = None
                return

            n_components = min(50, len(self.items) - 1, len(self.users) - 1)
            n_components = max(1, n_components)

            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd_model.fit(self.user_item_matrix)
            logger.info(
                f"Collaborative model trained with {n_components} components "
                f"({len(self.users)} users, {len(self.items)} items)"
            )
        except Exception as e:
            logger.error(f"Failed to build collaborative model: {e}")
            self.svd_model = None

    def recommend_for_lead(self, lead_id, top_n: int = 10):
        return self.recommend_for_user(lead_id, top_n=top_n)

    def recommend_for_user(self, user_id, top_n: int = 10):
        user_id = str(user_id)
        if self.svd_model is None or self.user_item_matrix is None:
            return []

        if user_id not in self.users:
            logger.info(f"Collaborative: user {user_id} not in matrix")
            return []

        try:
            user_idx = self.users.index(user_id)
            user_vector = self.svd_model.transform(
                self.user_item_matrix.iloc[[user_idx]]
            )
            predicted_scores = np.dot(user_vector, self.svd_model.components_).flatten()

            known = self.user_item_matrix.iloc[user_idx]
            known_items = set(known[known > 0].index.astype(str).tolist())

            sorted_indices = np.argsort(predicted_scores)[::-1]
            recommendations = []
            seen = set()

            for idx in sorted_indices:
                if len(recommendations) >= top_n:
                    break
                prop_id = str(self.items[idx])
                if prop_id in seen or prop_id in known_items:
                    continue
                score = float(predicted_scores[idx])
                if score <= 0.05:
                    continue
                recommendations.append({
                    "property_id": prop_id,
                    "score": round(score, 4),
                    "reason": "Similar clients also liked this property",
                    "source": "collaborative",
                })
                seen.add(prop_id)

            return recommendations
        except Exception as e:
            logger.error(f"Collaborative recommendation error: {e}")
            return []
