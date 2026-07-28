from flask import Flask, request, jsonify
from flask_cors import CORS
from .config import Config
from services.ai_services import AIServices
import pandas as pd
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommender.recommendation_service import RecommendationService
from recommender.user_stage import UserStageDetector


app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

if 'SQLALCHEMY_DATABASE_URI' not in app.config:
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ai_services = AIServices(Config)   
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

recommender_service = RecommendationService()
stage_detector = UserStageDetector()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'AI API is running', 'timestamp': datetime.now().isoformat()})


@app.route('/ai/predict/conversion', methods=['POST'])
def predict_conversion():
    data = request.get_json()
    if not data or 'lead_id' not in data:
        return jsonify({'error': 'lead_id required'}), 400
    lead_id = int(data['lead_id'])
    prob = ai_services.predict_conversion(lead_id)
    if prob is None:
        return jsonify({'error': 'Lead not found'}), 404
    
    with engine.connect() as conn:
        conn.execute(text(f"UPDATE leads SET conversion_probability = {prob} WHERE lead_id = {lead_id}"))
    return jsonify({'lead_id': lead_id, 'conversion_probability': prob})

@app.route('/ai/segment/lead', methods=['POST'])
def segment_lead():
    data = request.get_json()
    if not data or 'lead_id' not in data:
        return jsonify({'error': 'lead_id required'}), 400
    lead_id = int(data['lead_id'])
    segment = ai_services.segment_lead(lead_id)
    if segment is None:
        return jsonify({'error': 'Segment not found for this lead'}), 404
    return jsonify({'lead_id': lead_id, 'segment': segment})


@app.route('/ai/priority/leads', methods=['GET'])
def priority_leads():
    limit = request.args.get('limit', 10, type=int)
    query = f"""
    SELECT lead_id, budget, engagement_score, conversion_probability, priority_score
    FROM lead_priorities
    ORDER BY priority_score DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, engine)
    return jsonify(df.to_dict(orient='records'))


@app.route('/ai/recommend/properties', methods=['POST'])
def recommend_properties():
    data = request.get_json()
    if not data or 'user_id' not in data or 'user_type' not in data:
        return jsonify({'error': 'user_id and user_type (lead/client) required'}), 400
    user_id = int(data['user_id'])
    user_type = data['user_type']
    limit = data.get('limit', 10)
    
    result = recommender_service.recommend(user_id, top_n=limit)
    recommendations = result.get('recommendations', [])
    return jsonify({'user_id': user_id, 'user_type': user_type, 'recommendations': recommendations})

@app.route('/ai/recommend/employee', methods=['POST'])
def recommend_employee():
    data = request.get_json()
    if not data or 'deal_id' not in data:
        return jsonify({'error': 'deal_id required'}), 400
    deal_id = int(data['deal_id'])

    recs = ai_services.recommend_employees_for_deal(deal_id, top_n=5)
    if not recs:
        return jsonify({'error': 'No recommendations generated'}), 404
    

    for idx, rec in enumerate(recs, 1):
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO employee_deal_recommendations (deal_id, employee_id, score, rank)
                VALUES (:deal_id, :emp_id, :score, :rank)
                ON DUPLICATE KEY UPDATE score = VALUES(score), rank = VALUES(rank)
            """), {"deal_id": deal_id, "emp_id": rec['employee_id'], "score": rec['score'], "rank": idx})
    return jsonify({'deal_id': deal_id, 'recommendations': recs})

@app.route('/ai/forecast/sales', methods=['GET'])
def sales_forecast():
    months = request.args.get('months', 6, type=int)
    query = f"""
    SELECT month, predicted_leads, predicted_conversion_rate, predicted_conversions, predicted_revenue
    FROM sales_forecast
    ORDER BY month
    LIMIT {months}
    """
    df = pd.read_sql(query, engine)
    return jsonify(df.to_dict(orient='records'))


@app.route('/ai/user/stage/<int:user_id>', methods=['GET'])
def user_stage(user_id):
    stage = stage_detector.get_stage(user_id)
    return jsonify({'user_id': user_id, 'stage': stage})

if __name__ == '__main__':
    print("="*60)
    print(" AI Flask API is running on http://{}:{}".format(app.config['HOST'], app.config['PORT']))
    print(" Available endpoints:")
    print("  POST /ai/predict/conversion")
    print("  POST /ai/segment/lead")
    print("  GET  /ai/priority/leads")
    print("  POST /ai/recommend/properties")
    print("  POST /ai/recommend/employee")
    print("  GET  /ai/forecast/sales")
    print("  GET  /ai/user/stage/<user_id>")
    print("="*60)
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])
