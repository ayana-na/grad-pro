import pandas as pd
import mysql.connector
import joblib

print("Loading AI model...")

# تحميل الموديل
model = joblib.load("lead_model.pkl")

# تحميل ترتيب الأعمدة
columns = joblib.load("lead_model_columns.pkl")

print("Connecting to database...")

conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="real_estate_ai",
    use_pure=True
)

query = """
SELECT
    lead_id,
    budget,
    urgency_level,
    source,
    engagement_score,
    price_gap
FROM leads
"""

cursor = conn.cursor(dictionary=True)
cursor.execute(query)

rows = cursor.fetchall()

df = pd.DataFrame(rows)

lead_ids = df["lead_id"]

# تحويل categorical
df = pd.get_dummies(df, columns=["urgency_level", "source"])

# حذف id
df = df.drop("lead_id", axis=1)

# إعادة ترتيب الأعمدة مثل التدريب
df = df.reindex(columns=columns, fill_value=0)

print("Predicting lead conversion probability...")

probabilities = model.predict_proba(df)[:, 1]

# تحويل الاحتمال إلى score
scores = (probabilities * 100).clip(1, 99)

updates = []

for i, lead_id in enumerate(lead_ids):

    score = float(scores[i])

    # تصنيف lead
    if score >= 70:
        category = "HOT"
    elif score >= 40:
        category = "WARM"
    else:
        category = "COLD"

    updates.append((score, category, int(lead_id)))

update_query = """
UPDATE leads
SET conversion_probability = %s,
    lead_category = %s
WHERE lead_id = %s
"""

cursor.executemany(update_query, updates)

conn.commit()

cursor.close()
conn.close()

print("Lead scores updated successfully!")