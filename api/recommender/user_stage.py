from sqlalchemy import create_engine, text

class UserStageDetector:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")
    
    def get_stage(self, user_id: int):
        try:
            query = text("""
                SELECT 
                    CASE 
                        WHEN EXISTS (SELECT 1 FROM leads WHERE lead_id = :uid) THEN 'LEAD'
                        ELSE 'VISITOR'
                    END as stage,
                    COALESCE((SELECT COUNT(*) FROM leads WHERE lead_id = :uid), 0) as interactions
                FROM dual
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"uid": user_id}).fetchone()
            
            stage = result[0] if result else 'VISITOR'
            interactions = result[1] if result else 0

            if stage == 'LEAD':
                if interactions >= 8:
                    return 'HOT_LEAD'
                elif interactions >= 3:
                    return 'WARM_LEAD'
                else:
                    return 'LEAD'
            
            return stage
            
        except Exception as e:
            print(f"Stage detection error: {e}")
            return 'VISITOR'