from sqlalchemy import create_engine, text
import os


class UserStageDetector:
    def __init__(self, engine=None):
        if engine is not None:
            self.engine = engine
        else:
            try:
                from config import Config
                uri = Config.SQLALCHEMY_DATABASE_URI
            except Exception:
                uri = os.environ.get("DATABASE_URL", "")
                if uri.startswith("postgresql://"):
                    uri = uri.replace("postgresql://", "postgresql+psycopg2://", 1)
            self.engine = create_engine(uri, pool_pre_ping=True)

    def get_stage(self, user_id, user_type=None):
        hinted = None
        if user_type:
            t = str(user_type).strip().upper()
            if t in ("CLIENT", "LEAD", "VISITOR", "WARM_LEAD", "HOT_LEAD"):
                hinted = t

        uid = str(user_id)

        try:
            with self.engine.connect() as conn:
                client_row = conn.execute(
                    text("""
                        SELECT id
                        FROM clients
                        WHERE id::text = :uid
                           OR user_id::text = :uid
                        LIMIT 1
                    """),
                    {"uid": uid}
                ).fetchone()
                if client_row:
                    return "CLIENT"

                lead_row = conn.execute(
                    text("""
                        SELECT id
                        FROM leads
                        WHERE id::text = :uid
                        LIMIT 1
                    """),
                    {"uid": uid}
                ).fetchone()

                if not lead_row:
                    return "VISITOR"

                behavior = conn.execute(
                    text("""
                        SELECT
                            COUNT(*) AS total_events,
                            COUNT(*) FILTER (
                                WHERE event_type::text IN ('VIEW', 'LIKE')
                            ) AS views_or_likes,
                            COUNT(*) FILTER (
                                WHERE event_type::text = 'SHARE'
                            ) AS shares,
                            AVG(dwell_time) AS avg_dwell
                        FROM lead_behaviors
                        WHERE lead_id::text = :uid
                    """),
                    {"uid": uid}
                ).fetchone()

                if not behavior or int(behavior.total_events or 0) == 0:
                    return "LEAD"

                activity_score = (
                    float(behavior.views_or_likes or 0) * 0.15
                    + float(behavior.shares or 0) * 0.40
                    + float(behavior.avg_dwell or 0) / 60.0 * 0.20
                    + float(behavior.total_events or 0) * 0.05
                )

                if activity_score >= 5.0:
                    return "HOT_LEAD"
                if activity_score >= 2.5:
                    return "WARM_LEAD"
                return "LEAD"

        except Exception as e:
            print(f"Stage detection error: {e}")
            if hinted:
                return hinted
            return "VISITOR"
