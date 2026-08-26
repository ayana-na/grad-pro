import math
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from queries.lead_queries import (
    get_lead_conversion_data,
    get_lead_segment,
    update_lead_conversion_probability,
)
from queries.employee_queries import get_employees_by_type
from queries.request_queries import get_request_by_id
from queries.deal_queries import (
    get_employee_performance_sale_lease,
    get_employee_performance_buy_rent,
)
from queries.client_priority_queries import (
    get_clients_interested_in_property,
    get_open_request_clients,
)
from queries.property_queries import get_property_context
from queries.context_queries import get_client_context
from .feature_engineering import prepare_conversion_features

logger = logging.getLogger(__name__)


def _safe_float(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def _request_strength(row) -> float:
    status = str(row.get("request_status") or "").upper()
    if status == "IN_PROGRESS":
        base = 0.85
    elif status == "PENDING":
        base = 0.65
    elif row.get("request_id"):
        base = 0.50
    else:
        base = 0.25

    if row.get("has_schedule"):
        base = min(1.0, base + 0.15)
    return base


def _fit_score(row) -> float:
    price = _safe_float(row.get("listed_price"), None)
    min_b = row.get("preferred_budget_min")
    max_b = row.get("preferred_budget_max")
    score = 0.5

    if price is not None and min_b is not None and max_b is not None:
        min_b, max_b = float(min_b), float(max_b)
        if min_b <= price <= max_b:
            score = 1.0
        else:
            if price < min_b:
                gap = (min_b - price) / max(min_b, 1)
            else:
                gap = (price - max_b) / max(max_b, 1)
            score = max(0.0, 1.0 - gap)

    pref_rooms = row.get("preferred_rooms")
    rooms = row.get("num_of_rooms")
    if pref_rooms is not None and rooms is not None:
        try:
            if int(pref_rooms) == int(rooms):
                score = min(1.0, score + 0.1)
            elif abs(int(pref_rooms) - int(rooms)) == 1:
                score = min(1.0, score + 0.05)
        except Exception:
            pass

    pref_loc = (str(row.get("preferred_location") or "")).strip().lower()
    city = (str(row.get("city") or "")).strip().lower()
    location = (str(row.get("location") or "")).strip().lower()
    if pref_loc and (pref_loc in city or pref_loc in location or city in pref_loc):
        score = min(1.0, score + 0.1)

    return max(0.0, min(1.0, score))


def _pipeline_boost(row) -> float:
    status = str(row.get("request_status") or "").upper()
    if status == "IN_PROGRESS" and row.get("has_schedule"):
        return 1.0
    if status == "IN_PROGRESS":
        return 0.8
    if status == "PENDING":
        return 0.5
    return 0.2


def _deal_value_norm(row, max_price: float) -> float:
    price = _safe_float(row.get("listed_price"), 0.0)
    if max_price <= 0:
        return 0.0
    return max(0.0, min(1.0, price / max_price))


def _wait_factor(row) -> float:
    ts = row.get("request_created_at") or row.get("client_created_at")
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return 0.0
    try:
        if isinstance(ts, str):
            ts = pd.to_datetime(ts, utc=True)
        now = datetime.now(timezone.utc)
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        days = max(0.0, (now - ts).total_seconds() / 86400.0)
        return min(1.0, days / 14.0)
    except Exception:
        return 0.0


def _client_priority_reason(row, scores: dict) -> str:
    parts = []
    if scores["request_strength"] >= 0.8:
        parts.append("Active deal request")
    elif scores["request_strength"] >= 0.6:
        parts.append("Open request")
    else:
        parts.append("Favorited property")
    if scores["fit_score"] >= 0.85:
        parts.append("strong budget/location fit")
    if row.get("has_schedule"):
        parts.append("visit scheduled")
    return " + ".join(parts)


class AIServices:
    def __init__(self, config):
        self.config = config
        self.engine = create_engine(config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)

        self.conversion_model = None
        self.conversion_features = None
        self.employee_model = None
        self.employee_features = None
        self.employee_encoders = None

        try:
            self.conversion_model = joblib.load(config.CONVERSION_MODEL_PATH)
            self.conversion_features = joblib.load(config.CONVERSION_FEATURES_PATH)
            logger.info("Conversion model loaded")
        except Exception as e:
            logger.warning(f"Conversion model not loaded: {e}")

        try:
            self.employee_model = joblib.load(config.EMPLOYEE_MODEL_PATH)
            self.employee_features = joblib.load(config.EMPLOYEE_FEATURES_PATH)
            self.employee_encoders = joblib.load(config.EMPLOYEE_ENCODERS_PATH)
            logger.info("Employee model loaded")
        except Exception as e:
            logger.warning(f"Employee model not loaded: {e}")

    @staticmethod
    def _map_request_to_routing(request_type: str):
        t = (request_type or "").upper()
        if t == "SELL":
            return "SALES", "SALE_LEASE", "SALE"
        if t == "LEASE":
            return "LEASE", "SALE_LEASE", "LEASE"
        if t == "BUY":
            return "PURCHASING", "BUY_RENT", "BUY"
        if t == "RENT":
            return "RENT", "BUY_RENT", "LEASE"
        return "SALES", "SALE_LEASE", "SALE"

    def predict_conversion(self, lead_id):
        if self.conversion_model is None or self.conversion_features is None:
            logger.error("Conversion model missing")
            return None

        df = pd.read_sql(
            get_lead_conversion_data(),
            self.engine,
            params=(str(lead_id),),
        )
        if df.empty:
            return None

        df_feat = prepare_conversion_features(df)
        cols = [c for c in ["urgency_level", "source", "neighborhood"] if c in df_feat.columns]
        df_encoded = pd.get_dummies(df_feat, columns=cols, drop_first=True) if cols else df_feat

        for col in self.conversion_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        X = df_encoded[self.conversion_features]
        prob = float(self.conversion_model.predict_proba(X)[:, 1][0])
        return prob

    def update_conversion_probability(self, lead_id, probability: float):
        sql = update_lead_conversion_probability()
        with self.engine.begin() as conn:
            if ":lead_id" in sql or ":prob" in sql:
                conn.execute(
                    text(sql),
                    {"prob": float(probability), "lead_id": str(lead_id)},
                )
            else:
                conn.execute(text(sql), (float(probability), str(lead_id)))

    def segment_lead(self, lead_id):
        df = pd.read_sql(
            get_lead_segment(),
            self.engine,
            params=(str(lead_id),),
        )
        if df.empty:
            return None
        return df.iloc[0]["segment"]

    def recommend_employees_for_request(self, request_id, top_n: int = 5):
        request_id = str(request_id)
        req_df = pd.read_sql(
            get_request_by_id(),
            self.engine,
            params=(request_id,),
        )
        if req_df.empty:
            return None

        req = req_df.iloc[0].to_dict()
        return self.recommend_employees_for_context(
            request_type=str(req.get("request_type") or ""),
            property_id=req.get("property_id"),
            client_id=req.get("client_id"),
            top_n=top_n,
            request_id=request_id,
        )

    def recommend_employees_for_context(
        self,
        request_type: str,
        property_id=None,
        client_id=None,
        top_n: int = 5,
        request_id=None,
    ):
        request_type = str(request_type or "").upper()
        if request_type not in {"SELL", "BUY", "RENT", "LEASE"}:
            return {
                "error": "request_type must be one of SELL, BUY, RENT, LEASE",
                "count": 0,
                "candidates": [],
            }

        employee_type, deal_kind, perf_deal_type = self._map_request_to_routing(request_type)

        prop = {}
        if property_id:
            try:
                pdf = pd.read_sql(
                    get_property_context(),
                    self.engine,
                    params=(str(property_id),),
                )
                if not pdf.empty:
                    prop = pdf.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"property context load failed: {e}")

        client = {}
        if client_id:
            try:
                cdf = pd.read_sql(
                    get_client_context(),
                    self.engine,
                    params=(str(client_id),),
                )
                if not cdf.empty:
                    client = cdf.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"client context load failed: {e}")

        employees = pd.read_sql(
            get_employees_by_type(),
            self.engine,
            params=(employee_type,),
        )
        if employees.empty:
            return {
                "request_id": str(request_id) if request_id else None,
                "request_type": request_type,
                "deal_kind": deal_kind,
                "performance_deal_type": perf_deal_type,
                "employee_type": employee_type,
                "property_id": str(property_id) if property_id else None,
                "client_id": str(client_id) if client_id else None,
                "count": 0,
                "candidates": [],
                "message": f"No active employees with type={employee_type}",
            }

        if deal_kind == "SALE_LEASE":
            perf = pd.read_sql(
                get_employee_performance_sale_lease(),
                self.engine,
                params=(perf_deal_type,),
            )
        else:
            perf = pd.read_sql(
                get_employee_performance_buy_rent(),
                self.engine,
                params=(perf_deal_type,),
            )

        employees["employee_id"] = employees["employee_id"].astype(str)
        if perf is not None and not perf.empty:
            perf["employee_id"] = perf["employee_id"].astype(str)
            employees = employees.merge(perf, on="employee_id", how="left")
        else:
            employees["deals_won"] = 0
            employees["deals_lost"] = 0
            employees["total_closed"] = 0
            employees["avg_profit"] = None

        for col in ["deals_won", "deals_lost", "total_closed"]:
            employees[col] = employees[col].fillna(0).astype(float)

        employees["win_rate"] = employees["deals_won"] / (employees["total_closed"] + 1e-9)

        prop_city = str(prop.get("city") or prop.get("location") or "").lower()
        pref_loc_client = str(client.get("preferred_location") or "").lower()
        prop_type = str(prop.get("property_type") or "").lower()

        candidates = []
        for _, emp in employees.iterrows():
            win_rate = float(emp["win_rate"])
            productivity = float(emp.get("productivity") or 0.5)
            if productivity > 1.5:
                productivity = productivity / 100.0

            loc_emp = str(emp.get("preferred_location") or emp.get("location") or "").lower()
            location_fit = 0.0
            if loc_emp and prop_city and (loc_emp in prop_city or prop_city in loc_emp):
                location_fit = 1.0
            elif loc_emp and pref_loc_client and (
                loc_emp in pref_loc_client or pref_loc_client in loc_emp
            ):
                location_fit = 0.8

            emp_pref_type = str(emp.get("preferred_property_type") or "").lower()
            type_fit = 1.0 if emp_pref_type and prop_type and emp_pref_type == prop_type else 0.0

            response = float(emp.get("average_response_time") or 24.0)
            response_score = max(0.0, min(1.0, 1.0 - (response / 48.0)))

            score = (
                0.40 * win_rate
                + 0.25 * productivity
                + 0.20 * location_fit
                + 0.10 * type_fit
                + 0.05 * response_score
            )

            avg_profit = emp.get("avg_profit")
            if avg_profit is not None and not pd.isna(avg_profit) and float(avg_profit) > 0:
                score = min(1.0, score + 0.05)

            reasons = []
            if emp["total_closed"] == 0:
                reasons.append("limited history")
            elif win_rate >= 0.6:
                reasons.append("strong close rate")
            else:
                reasons.append("moderate close rate")
            if location_fit >= 0.8:
                reasons.append("location fit")
            if type_fit >= 1.0:
                reasons.append("property type fit")

            candidates.append({
                "employee_id": str(emp["employee_id"]),
                "name": emp.get("full_name") or emp.get("name"),
                "employee_type": employee_type,
                "score": round(float(score), 4),
                "win_rate": round(win_rate, 4),
                "deals_won": int(emp["deals_won"]),
                "deals_lost": int(emp["deals_lost"]),
                "total_closed": int(emp["total_closed"]),
                "reason": ", ".join(reasons),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[: int(top_n)]

        return {
            "request_id": str(request_id) if request_id else None,
            "request_type": request_type,
            "deal_kind": deal_kind,
            "performance_deal_type": perf_deal_type,
            "property_id": str(property_id) if property_id else None,
            "client_id": str(client_id) if client_id else None,
            "employee_type": employee_type,
            "count": len(top),
            "candidates": top,
        }

    def prioritize_clients_for_property(self, property_id, limit: int = 10):
        property_id = str(property_id)
        df = pd.read_sql(
            get_clients_interested_in_property(),
            self.engine,
            params=(property_id,),
        )
        if df.empty:
            return {"property_id": property_id, "count": 0, "candidates": []}

        max_price = float(df["listed_price"].fillna(0).max() or 1.0)
        candidates = []

        for _, row in df.iterrows():
            row = row.to_dict()
            s_req = _request_strength(row)
            s_fit = _fit_score(row)
            s_pipe = _pipeline_boost(row)
            s_val = _deal_value_norm(row, max_price)
            s_wait = _wait_factor(row)
            priority = (
                0.30 * s_req
                + 0.25 * s_fit
                + 0.20 * s_pipe
                + 0.15 * s_val
                + 0.10 * s_wait
            )
            scores = {
                "request_strength": round(s_req, 3),
                "fit_score": round(s_fit, 3),
                "pipeline_boost": round(s_pipe, 3),
                "deal_value_norm": round(s_val, 3),
                "wait_factor": round(s_wait, 3),
            }
            candidates.append({
                "client_id": str(row.get("client_id")),
                "client_name": row.get("client_name"),
                "phone": row.get("phone"),
                "request_id": str(row["request_id"]) if row.get("request_id") else None,
                "request_type": row.get("request_type"),
                "request_status": row.get("request_status"),
                "property_id": property_id,
                "listed_price": _safe_float(row.get("listed_price"), None),
                "priority_score": round(priority, 4),
                "scores": scores,
                "expected_deal_value": _safe_float(row.get("listed_price"), 0.0),
                "reason": _client_priority_reason(row, scores),
            })

        candidates.sort(key=lambda x: x["priority_score"], reverse=True)
        return {
            "property_id": property_id,
            "count": len(candidates[:limit]),
            "candidates": candidates[:limit],
        }

    def prioritize_open_client_requests(self, limit: int = 20):
        df = pd.read_sql(
            get_open_request_clients(),
            self.engine,
            params=(max(limit * 3, limit),),
        )
        if df.empty:
            return {"count": 0, "candidates": []}

        max_price = float(df["listed_price"].fillna(0).max() or 1.0)
        candidates = []

        for _, row in df.iterrows():
            row = row.to_dict()
            s_req = _request_strength(row)
            s_fit = _fit_score(row)
            s_pipe = _pipeline_boost(row)
            s_val = _deal_value_norm(row, max_price)
            s_wait = _wait_factor(row)
            priority = (
                0.30 * s_req
                + 0.25 * s_fit
                + 0.20 * s_pipe
                + 0.15 * s_val
                + 0.10 * s_wait
            )
            scores = {
                "request_strength": round(s_req, 3),
                "fit_score": round(s_fit, 3),
                "pipeline_boost": round(s_pipe, 3),
                "deal_value_norm": round(s_val, 3),
                "wait_factor": round(s_wait, 3),
            }
            candidates.append({
                "client_id": str(row.get("client_id")),
                "client_name": row.get("client_name"),
                "phone": row.get("phone"),
                "request_id": str(row.get("request_id")) if row.get("request_id") else None,
                "request_type": row.get("request_type"),
                "request_status": row.get("request_status"),
                "property_id": str(row["property_id"]) if row.get("property_id") else None,
                "priority_score": round(priority, 4),
                "scores": scores,
                "reason": _client_priority_reason(row, scores),
            })

        candidates.sort(key=lambda x: x["priority_score"], reverse=True)
        return {
            "count": len(candidates[:limit]),
            "candidates": candidates[:limit],
        }

    def extract_client_insights(self, text=None, messages=None):
        from .client_insights import extract_from_text, extract_from_messages

        if messages and isinstance(messages, list):
            return extract_from_messages(messages)
        if text and str(text).strip():
            return extract_from_text(str(text))
        return {
            "preferences": [],
            "avoidances": [],
            "error": "text or messages required",
        }
