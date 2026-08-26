import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    _raw_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://neondb_owner:npg_kcqTr5H4YJzF@ep-steep-tree-aif750w5-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )

    if "channel_binding=" in _raw_url:
        parts = _raw_url.split("?")
        if len(parts) == 2:
            base, query = parts
            params = [p for p in query.split("&") if not p.startswith("channel_binding=")]
            _raw_url = base + (("?" + "&".join(params)) if params else "")

    if _raw_url.startswith("postgresql://"):
        SQLALCHEMY_DATABASE_URI = _raw_url.replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
    elif _raw_url.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = _raw_url.replace(
            "postgres://", "postgresql+psycopg2://", 1
        )
    else:
        SQLALCHEMY_DATABASE_URI = _raw_url

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    CONVERSION_MODEL_PATH = os.path.join(MODEL_DIR, "conversion_model_optimized.pkl")
    CONVERSION_FEATURES_PATH = os.path.join(MODEL_DIR, "conversion_features_optimized.pkl")
    EMPLOYEE_MODEL_PATH = os.path.join(MODEL_DIR, "employee_matcher_xgb.pkl")
    EMPLOYEE_FEATURES_PATH = os.path.join(MODEL_DIR, "employee_matcher_features.pkl")
    EMPLOYEE_ENCODERS_PATH = os.path.join(MODEL_DIR, "employee_matcher_encoders.pkl")

    PORT = int(os.environ.get("PORT", 5001))
    HOST = "0.0.0.0"
    DEBUG = False
