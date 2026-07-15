from sklearn.datasets import fetch_california_housing
import mysql.connector
import random
import pandas as pd

print("STARTING LEAD GENERATION...")

# ---------------------------
# 1. تحميل dataset العقارات
# ---------------------------
data = fetch_california_housing(as_frame=True)
df = data.frame

# ---------------------------
# 2. الاتصال بقاعدة البيانات
# ---------------------------
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="real_estate_ai"
)

cursor = conn.cursor()

print("Connected to database")

# ---------------------------
# 3. جلب property_id الحقيقي
# ---------------------------
cursor.execute("SELECT property_id, price FROM properties")
properties = cursor.fetchall()

print("Number of properties:", len(properties))

# ---------------------------
# 4. توزيع مصادر العملاء
# ---------------------------
SOURCES = ['Website', 'Referral', 'Zillow', 'Facebook', 'Instagram', 'Realtor']
SOURCE_WEIGHTS = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]

URGENCY_LEVELS = ['low', 'medium', 'high']
URGENCY_WEIGHTS = [0.30, 0.50, 0.20]

leads_created = 0

# ---------------------------
# 5. توليد Leads
# ---------------------------
for prop in properties:

    property_id = prop[0]
    price = float(prop[1])

    # ليس كل عقار ينتج lead
    if random.random() > 0.4:
        continue

    # -------------------
    # الميزانية
    # -------------------
    rand = random.random()

    if rand < 0.70:
        budget_factor = random.uniform(0.85, 1.15)

    elif rand < 0.90:
        budget_factor = random.uniform(0.60, 0.84)

    else:
        budget_factor = random.uniform(1.16, 1.40)

    budget = round(price * budget_factor, 2)

    # -------------------
    # مصدر العميل
    # -------------------
    source = random.choices(SOURCES, weights=SOURCE_WEIGHTS)[0]

    # -------------------
    # الاستعجال
    # -------------------
    urgency = random.choices(URGENCY_LEVELS, weights=URGENCY_WEIGHTS)[0]

    # -------------------
    # engagement score
    # -------------------
    base_engagement = 0.5

    if source in ['Referral', 'Zillow']:
        base_engagement += 0.15

    if source in ['Website', 'Realtor']:
        base_engagement += 0.05

    if urgency == 'high':
        base_engagement += 0.20

    elif urgency == 'medium':
        base_engagement += 0.10

    engagement_score = random.uniform(base_engagement - 0.1, base_engagement + 0.1)
    engagement_score = round(min(1.0, max(0.2, engagement_score)), 2)

    # -------------------
    # price gap
    # -------------------
    price_gap = round(budget - price, 2)

    # -------------------
    # conversion probability
    # -------------------
    prob = engagement_score * 0.5

    if price_gap > 0:
        prob += 0.2

    elif price_gap < -20000:
        prob -= 0.15

    if source in ['Referral', 'Zillow']:
        prob += 0.1

    conversion_probability = round(min(0.98, max(0.05, prob)), 2)

    # -------------------
    # هل تم التحويل
    # -------------------
    conversion_status = 1 if random.random() < conversion_probability else 0

    # -------------------
    # agent
    # -------------------
    assigned_agent = None

    # -------------------
    # إدخال البيانات
    # -------------------
    cursor.execute("""
        INSERT INTO leads
        (property_id, budget, urgency_level, source, assigned_agent,
        engagement_score, price_gap, conversion_probability, conversion_status)

        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        property_id,
        budget,
        urgency,
        source,
        assigned_agent,
        engagement_score,
        price_gap,
        conversion_probability,
        conversion_status
    ))

    leads_created += 1

# ---------------------------
# حفظ البيانات
# ---------------------------
conn.commit()

cursor.close()
conn.close()

print("Leads created:", leads_created)
print("FINISHED SUCCESSFULLY")