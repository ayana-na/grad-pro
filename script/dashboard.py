import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import plotly.express as px
import os
from datetime import datetime

import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

st.set_page_config(page_title="Real Estate AI CRM", layout="wide")

st.title(" AI Real Estate CRM Dashboard")
st.markdown("---")


@st.cache_resource
def get_engine():
    return create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")

engine = get_engine()


@st.cache_data
def load_counts():
    with engine.connect() as conn:
        properties = conn.execute(text("SELECT COUNT(*) FROM properties")).scalar()
        leads = conn.execute(text("SELECT COUNT(*) FROM leads")).scalar()
    return properties, leads


@st.cache_data
def load_segments():
    return pd.read_sql("SELECT segment, COUNT(*) as total FROM customer_segments GROUP BY segment", engine)


@st.cache_data
def load_priorities():
    return pd.read_sql("SELECT * FROM lead_priorities ORDER BY priority_score DESC LIMIT 10", engine)


@st.cache_data
def load_revenue():
    return pd.read_sql("SELECT * FROM revenue_leads ORDER BY expected_revenue DESC LIMIT 10", engine)


@st.cache_data
def load_forecast():
    return pd.read_sql("SELECT * FROM sales_forecast ORDER BY month", engine)


props, leads = load_counts()

col1, col2 = st.columns(2)

col1.metric(" Properties", f"{props:,}")
col2.metric(" Leads", f"{leads:,}")

st.markdown("---")


st.subheader(" Customer Segmentation")
segments = load_segments()
fig = px.pie(segments, values="total", names="segment", title="Lead Segments Distribution")
st.plotly_chart(fig, use_container_width=True)


st.subheader(" Top Priority Leads")
priorities = load_priorities()
st.dataframe(
    priorities[["lead_id", "conversion_probability", "engagement_score", "priority_score"]],
    use_container_width=True
)


st.subheader(" Top Revenue Opportunities")
revenue = load_revenue()
st.dataframe(
    revenue[["lead_id", "budget", "conversion_probability", "expected_revenue"]],
    use_container_width=True
)


st.subheader(" Sales Forecast")
forecast = load_forecast()
if not forecast.empty:
    forecast["month"] = pd.to_datetime(forecast["month"])
    fig = px.line(
        forecast,
        x="month",
        y="predicted_conversions",
        title="Predicted Conversions",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Forecast Data"):
        st.dataframe(forecast, use_container_width=True)
else:
    st.info("Run sales_forecast.py to generate forecast")


st.markdown("---")
st.subheader("Model Explainability (Conversion Probability)")


@st.cache_resource
def load_conversion_model_and_features():
    try:
        pipeline = joblib.load("models/conversion_model_optimized.pkl")
        features = joblib.load("models/conversion_features_optimized.pkl")
        

        possible_keys = ['model', 'classifier', 'clf', 'catboost', 'estimator']
        catboost_model = None
        for key in possible_keys:
            if key in pipeline.named_steps:
                catboost_model = pipeline.named_steps[key]
                break
        
        if catboost_model is None:
            raise ValueError("Could not find the classifier step in the pipeline")
            
        return catboost_model, features
    except Exception as e:
        st.error(f"Failed to load/extract model: {str(e)}")
        return None, None


model, conversion_features = load_conversion_model_and_features()

if model is None or conversion_features is None:
    st.warning("Could not load conversion model or features. Run train_conversion_model.py first.")
else:
    @st.cache_resource
    def get_shap_explainer(_model):
        return shap.TreeExplainer(_model)

    explainer = get_shap_explainer(model)

    @st.cache_data
    def load_leads_for_shap():
        query = """
        SELECT 
            l.lead_id,
            l.budget           AS price,
            p.sqft,
            p.bedrooms,
            p.bathrooms,
            DATEDIFF(CURDATE(), p.listing_date) AS days_on_market,
            p.overall_qual,
            p.overall_cond,
            p.year_built,
            p.garage_cars,
            p.bsmt_sf,
            p.fireplaces,
            p.deck_sf,
            l.engagement_score,
            l.price_gap,
            p.neighborhood
        FROM leads l
        JOIN properties p ON l.property_id = p.property_id
        WHERE l.conversion_probability IS NOT NULL
        LIMIT 500
        """
        df = pd.read_sql(query, engine)

        if df.empty:
            return df

        df['price_per_sqft']        = df['price'] / df['sqft']
        df['property_age']          = 2026 - df['year_built']
        df['quality_index']         = df['overall_qual'] * df['overall_cond']
        df['luxury_score']          = (df['garage_cars'] * 2) + (df['fireplaces'] * 1.5) + (df['deck_sf'] / 100)
        df['price_to_income_ratio'] = df['price'] / (df['engagement_score'] * 100000 + 1)
        df['days_on_market_log']    = np.log1p(df['days_on_market'])
        df['sqft_per_bedroom']      = df['sqft'] / (df['bedrooms'] + 1)
        df['bath_per_bedroom']      = df['bathrooms'] / (df['bedrooms'] + 1)
        df['has_basement']          = (df['bsmt_sf'] > 0).astype(int)
        df['has_fireplace']         = (df['fireplaces'] > 0).astype(int)
        df['has_garage']            = (df['garage_cars'] > 0).astype(int)


        df = pd.get_dummies(df, columns=['neighborhood'], prefix='neighborhood', drop_first=True)

        df = df.reindex(columns=conversion_features, fill_value=0)


        df['lead_id'] = df.index if 'lead_id' not in df.columns else df['lead_id']

        return df

    df_shap = load_leads_for_shap()

    if df_shap.empty:
        st.info("No data available for SHAP explanation.")
    else:
        st.markdown("**Global Feature Importance**")

        if st.button("Compute & Show Summary Plot", key="global_shap_btn"):
            with st.spinner("Calculating SHAP values..."):
                try:
                    n = min(80, len(df_shap))
                    background = shap.kmeans(df_shap[conversion_features], n).data
                    shap_values = explainer.shap_values(background)

                    col_left, col_right = st.columns(2)

                    with col_left:
                        fig_bar = plt.figure(figsize=(9, 7))
                        shap.summary_plot(shap_values, background, plot_type="bar", show=False)
                        st.pyplot(fig_bar)
                        plt.close(fig_bar)

                    with col_right:
                        fig_detail = plt.figure(figsize=(9, 7))
                        shap.summary_plot(shap_values, background, show=False)
                        st.pyplot(fig_detail)
                        plt.close(fig_detail)

                except Exception as e:
                    st.error(f"Error computing global SHAP: {str(e)}")

        st.markdown("**Explain individual lead**")

        lead_list = ["None"] + df_shap["lead_id"].astype(str).tolist()
        selected = st.selectbox("Select a lead ID", lead_list)

        if selected != "None":
            try:
                lead_id = int(selected)
                row = df_shap[df_shap["lead_id"] == lead_id]

                if row.empty:
                    st.warning("Selected lead not found.")
                else:
                    X_instance = row[conversion_features].values

                    with st.spinner("Computing explanation..."):
                        shap_values_instance = explainer.shap_values(X_instance)

                       
                        if isinstance(shap_values_instance, list) and len(shap_values_instance) > 1:
                            sv = shap_values_instance[1][0]
                            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                        else:
                            sv = shap_values_instance[0] if shap_values_instance.ndim > 1 else shap_values_instance
                            base_val = explainer.expected_value

                        exp = shap.Explanation(
                            values=sv,
                            base_values=base_val,
                            data=X_instance[0],
                            feature_names=conversion_features
                        )

                        fig_w = plt.figure(figsize=(10, 6))
                        shap.plots.waterfall(exp, max_display=15, show=False)
                        st.pyplot(fig_w)
                        plt.close(fig_w)

                        prob = model.predict_proba(X_instance)[0, 1]
                        st.metric("Predicted Conversion Probability", f"{prob:.3%}")

            except Exception as e:
                st.error(f"Error explaining this lead: {str(e)}")


st.markdown("---")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
