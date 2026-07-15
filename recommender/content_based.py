# content_based.py
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self, properties_df: pd.DataFrame):
        self.properties_df = properties_df.copy()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.vectorizer = None
        self._build_content_model()

    def _build_content_model(self):
        """بناء نموذج المحتوى بطريقة آمنة"""
        try:
            df = self.properties_df.copy()

            # تحويل كل الأعمدة الرقمية إلى نص بأمان
            df['bedrooms_str'] = df['bedrooms'].fillna(0).astype(int).astype(str) + " غرفة"
            df['bathrooms_str'] = df['bathrooms'].fillna(0).astype(float).astype(str) + " حمام"
            df['sqft_str'] = df['sqft'].fillna(0).astype(int).astype(str) + " قدم مربع"
            df['qual_str'] = df['overall_qual'].fillna(0).astype(int).astype(str) + " جودة"

            # إنشاء عمود المحتوى النهائي
            df['content'] = (
                df.get('neighborhood', '').fillna('') + " " +
                df['bedrooms_str'] + " " +
                df['bathrooms_str'] + " " +
                df['qual_str'] + " " +
                df['sqft_str']
            ).str.strip()

            # في حال وجود قيم فارغة
            df['content'] = df['content'].replace('', 'عقار سكني عادي')

            # بناء TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(df['content'])
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
            # حفظ النسخة المعدلة
            self.properties_df = df

            logger.info(f"✅ تم بناء نموذج Content-Based بنجاح. عدد العقارات: {len(df)}")

        except Exception as e:
            logger.error(f"❌ خطأ في بناء نموذج Content-Based: {e}")
            self.similarity_matrix = None

    def recommend_for_lead(self, lead_id: int, top_n: int = 8) -> List[Dict]:
        """توصيات قائمة على المحتوى"""
        try:
            if self.similarity_matrix is None:
                logger.warning("نموذج Content-Based غير متاح")
                return []

            # جلب عقار مرتبط بالـ Lead
            query = f"""
                SELECT property_id 
                FROM leads 
                WHERE lead_id = {lead_id} 
                LIMIT 1
            """
            df = pd.read_sql(query, self.properties_df.engine if hasattr(self.properties_df, 'engine') else engine)
            if df.empty:
                return []

            source_property_id = int(df.iloc[0]['property_id'])

            if source_property_id not in self.properties_df['property_id'].values:
                return []

            idx = self.properties_df[self.properties_df['property_id'] == source_property_id].index[0]
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            recommendations = []
            for i, score in sim_scores[1:top_n+1]:
                prop = self.properties_df.iloc[i]
                recommendations.append({
                    'property_id': int(prop['property_id']),
                    'price': float(prop.get('price', 0)),
                    'sqft': float(prop.get('sqft', 0)),
                    'score': float(score),
                    'reason': f"يشبه العقار الذي تفاعلت معه (درجة تشابه: {score:.3f})"
                })

            return recommendations

        except Exception as e:
            logger.error(f"خطأ في توصيات Content-Based للـ Lead {lead_id}: {e}")
            return []