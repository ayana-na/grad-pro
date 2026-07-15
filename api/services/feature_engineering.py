import pandas as pd
import numpy as np

def prepare_conversion_features(data_df):
    """تطبق خطوات هندسة الميزات المستخدمة في training"""
    df = data_df.copy()
    df["price_per_sqft"] = df["budget"] / df["sqft"]
    df["property_age"] = 2026 - df["year_built"]
    df["quality_index"] = df["overall_qual"] * df["overall_cond"]
    df["luxury_score"] = (df["garage_cars"] * 2) + (df["fireplaces"] * 1.5) + (df["deck_sf"] / 100)
    df["price_to_income_ratio"] = df["budget"] / (df["engagement_score"] * 100000 + 1)
    df["days_on_market_log"] = np.log1p(df["days_on_market"])
    df["sqft_per_bedroom"] = df["sqft"] / (df["bedrooms"] + 1)
    df["bath_per_bedroom"] = df["bathrooms"] / (df["bedrooms"] + 1)
    df["has_basement"] = (df["bsmt_sf"] > 0).astype(int)
    df["has_fireplace"] = (df["fireplaces"] > 0).astype(int)
    df["has_garage"] = (df["garage_cars"] > 0).astype(int)
    return df