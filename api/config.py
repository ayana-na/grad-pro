import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', "mysql+pymysql://root:@127.0.0.1/real_estate_ai")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models") 
    CONVERSION_MODEL_PATH = os.path.join(MODEL_DIR, "conversion_model_optimized.pkl")
    CONVERSION_FEATURES_PATH = os.path.join(MODEL_DIR, "conversion_features_optimized.pkl")
    EMPLOYEE_MODEL_PATH = os.path.join(MODEL_DIR, "employee_matcher_xgb.pkl")
    EMPLOYEE_FEATURES_PATH = os.path.join(MODEL_DIR, "employee_matcher_features.pkl")
    EMPLOYEE_ENCODERS_PATH = os.path.join(MODEL_DIR, "employee_matcher_encoders.pkl")
    PORT = int(os.environ.get('PORT', 5001))
    HOST = '0.0.0.0'
    DEBUG = False
