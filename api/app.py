from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from services.ai_services import AIServices
from recommender.recommendation_service import RecommendationService
from recommender.user_stage import UserStageDetector
from queries.lead_queries import get_priority_leads
from queries.forecast_queries import get_sales_forecast
from sqlalchemy import create_engine
import pandas as pd
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ai_services = AIServices(Config)
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
recommender_service = RecommendationService(engine=engine)
stage_detector = UserStageDetector(engine=engine)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "AI API is running",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/ai/predict/conversion", methods=["POST"])
def predict_conversion():
    data = request.get_json()
    if not data or "lead_id" not in data:
        return jsonify({"error": "lead_id required"}), 400

    lead_id = data["lead_id"]
    prob = ai_services.predict_conversion(lead_id)

    if prob is None:
        return jsonify({"error": "Lead not found"}), 404

    try:
        ai_services.update_conversion_probability(lead_id, prob)
    except Exception as e:
        logger.warning(f"Could not update conversion probability: {e}")

    return jsonify({
        "lead_id": str(lead_id),
        "conversion_probability": prob
    })


@app.route("/ai/segment/lead", methods=["POST"])
def segment_lead():
    data = request.get_json()
    if not data or "lead_id" not in data:
        return jsonify({"error": "lead_id required"}), 400

    lead_id = data["lead_id"]
    segment = ai_services.segment_lead(lead_id)

    if segment is None:
        return jsonify({"error": "Segment not found for this lead"}), 404

    return jsonify({
        "lead_id": str(lead_id),
        "segment": segment
    })


@app.route("/ai/priority/leads", methods=["GET"])
def priority_leads():
    limit = request.args.get("limit", 10, type=int)
    try:
        df = pd.read_sql(get_priority_leads(), engine, params=(limit,))
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Priority leads error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai/recommend/properties", methods=["POST"])
def recommend_properties():
    data = request.get_json() or {}
    if "user_id" not in data:
        return jsonify({"error": "user_id required"}), 400

    user_id = data["user_id"]
    limit = int(data.get("limit", 10))
    user_type = data.get("user_type")

    try:
        result = recommender_service.recommend(
            user_id,
            top_n=limit,
            user_type=user_type,
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai/recommend/employee", methods=["POST"])
def recommend_employee():
    data = request.get_json() or {}
    top_n = int(data.get("top_n", 5))

    if data.get("request_id"):
        try:
            result = ai_services.recommend_employees_for_request(
                data["request_id"], top_n=top_n
            )
            if result is None:
                return jsonify({"error": "Request not found"}), 404
            return jsonify(result)
        except Exception as e:
            logger.error(f"Employee recommend error: {e}")
            return jsonify({"error": str(e)}), 500

    request_type = data.get("request_type")
    if not request_type:
        return jsonify({
            "error": "request_type is required (or send request_id)"
        }), 400

    try:
        result = ai_services.recommend_employees_for_context(
            request_type=request_type,
            property_id=data.get("property_id"),
            client_id=data.get("client_id"),
            top_n=top_n,
        )
        if result.get("error"):
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        logger.error(f"Employee recommend error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai/forecast/sales", methods=["GET"])
def sales_forecast():
    months = request.args.get("months", 6, type=int)
    try:
        df = pd.read_sql(get_sales_forecast(), engine, params=(months,))
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Sales forecast error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai/user/stage/<user_id>", methods=["GET"])
def user_stage(user_id):
    try:
        stage = stage_detector.get_stage(user_id)
        return jsonify({
            "user_id": str(user_id),
            "stage": stage
        })
    except Exception as e:
        logger.error(f"User stage error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai/priority/clients/by-property/<property_id>", methods=["GET"])
def priority_clients_by_property(property_id):
    limit = request.args.get("limit", 10, type=int)
    try:
        result = ai_services.prioritize_clients_for_property(property_id, limit=limit)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Client property priority error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai/priority/clients", methods=["GET"])
def priority_clients():
    limit = request.args.get("limit", 20, type=int)
    try:
        result = ai_services.prioritize_open_client_requests(limit=limit)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Client priority error: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/ai/client-insights/extract", methods=["POST"])
def client_insights_extract():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    messages = data.get("messages")

    if not text and not messages:
        return jsonify({"error": "text or messages required"}), 400

    try:
        result = ai_services.extract_client_insights(
            text=text,
            messages=messages,
        )
        if result.get("error"):
            return jsonify(result), 400
        return jsonify(result), 200
    except Exception as e:
        logger.exception("client insights extract failed")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    print("=" * 60)
    print(f" AI Flask API is running on http://{app.config.get('HOST', '0.0.0.0')}:{app.config.get('PORT', 5001)}")
    print(" Available endpoints:")
    print("  GET  /health")
    print("  POST /ai/predict/conversion")
    print("  POST /ai/segment/lead")
    print("  GET  /ai/priority/leads")
    print("  POST /ai/recommend/properties")
    print("  POST /ai/recommend/employee")
    print("  GET  /ai/forecast/sales")
    print("  GET  /ai/user/stage/<user_id>")
    print("  GET  /ai/priority/clients")
    print("  GET  /ai/priority/clients/by-property/<property_id>")
    print("=" * 60)

    app.run(
        host=app.config.get("HOST", "0.0.0.0"),
        port=app.config.get("PORT", 5001),
        debug=app.config.get("DEBUG", False)
    )
