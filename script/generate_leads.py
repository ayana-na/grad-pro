import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import random
from datetime import datetime, timedelta

print("="*60)
print(" Smart Lead Generator")
print("="*60)


engine = create_engine("mysql+pymysql://root:@127.0.0.1/real_estate_ai")



query = """
SELECT property_id, price, overall_qual, sqft, bedrooms,
       bathrooms, year_built, garage_cars, fireplaces
FROM properties
"""

df_prop = pd.read_sql(query, engine)

np.random.seed(42)

leads = []


for idx, row in df_prop.iterrows():

   
    if random.random() > 0.65:
        continue

    property_id = int(row["property_id"])
    price = float(row["price"])
    quality = int(row["overall_qual"])
    sqft = float(row["sqft"])

    bedrooms = int(row["bedrooms"])
    bathrooms = float(row["bathrooms"])
    year_built = int(row["year_built"])

    garage = int(row["garage_cars"] if pd.notna(row["garage_cars"]) else 0)
    fireplaces = int(row["fireplaces"] if pd.notna(row["fireplaces"]) else 0)


    budget = round(price * np.random.uniform(0.75,1.35),2)

    engagement = round(np.random.beta(2,1.5),2)

    price_gap = budget - price

    urgency = np.random.choice(
        ["low","medium","high"],
        p=[0.25,0.50,0.25]
    )

    source = np.random.choice(
        ["Website","Referral","Facebook","Instagram","Google"],
        p=[0.35,0.20,0.20,0.15,0.10]
    )

   

    agent_skill = np.random.uniform(0.4,1.0)

    

    base = (
        0.35*(engagement**2) +
        0.20*(quality/10) +
        0.10*(sqft/2500)
    )

   

    price_effect = 0.20*np.tanh(price_gap/20000)

    

    interaction = 0

    if quality > 7 and engagement > 0.7:
        interaction += 0.12

    if bedrooms >= 4 and bathrooms >= 2.5:
        interaction += 0.08

    if year_built > 2005:
        interaction += 0.05

    if sqft > 2200 and price_gap > 0:
        interaction += 0.10

    

    source_effect = {
        "Referral":0.12,
        "Website":0.08,
        "Google":0.07,
        "Instagram":0.05,
        "Facebook":0.04
    }[source]

    

    urgency_effect = {
        "low":-0.05,
        "medium":0.0,
        "high":0.10
    }[urgency]

    

    noise = np.random.normal(0,0.07)

    conversion_prob = (
        base +
        price_effect +
        interaction +
        source_effect +
        urgency_effect +
        0.10*agent_skill +
        noise
    )

    conversion_prob = np.clip(conversion_prob,0.05,0.95)

    conversion_status = 1 if np.random.random() < conversion_prob else 0

    

    lead = {

        "property_id":property_id,
        "budget":budget,
        "urgency_level":urgency,
        "source":source,
        "assigned_agent":None,
        "engagement_score":engagement,
        "price_gap":round(price_gap,2),
        "conversion_probability":round(conversion_prob,3),
        "conversion_status":conversion_status,
        "created_at":datetime.now() - timedelta(days=np.random.randint(1,90)),
        "lead_category":None,
        "updated_at":None
    }

    leads.append(lead)




if leads:

    leads_df = pd.DataFrame(leads)

    leads_df.to_sql(
        "leads",
        engine,
        if_exists="append",
        index=False
    )

    print(f"\n Generated {len(leads_df)} leads")

    print("\n Conversion Distribution:")

    print(leads_df["conversion_status"].value_counts())

else:

    print(" No leads generated")
