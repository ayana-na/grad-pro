import random
import mysql.connector
from mysql.connector import errorcode

print("SCRIPT IS RUNNING")
print("VERSION 3")
print("Connecting...!")

conn = None
cursor = None

try:
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="real_estate_ai",
        connection_timeout=10,
        use_pure=True
    )
    print("Connected successfully!")

    cursor = conn.cursor()
    cursor.execute("SELECT property_id, price FROM properties")
    properties = cursor.fetchall()

    print("Total properties found:", len(properties))

    if len(properties) == 0:
        print("No properties found!")
        exit()

    leads_created = 0

    for prop in properties:
        property_id = prop[0]
        price = float(prop[1])

        if random.random() < 0.3:
            budget = round(price * random.uniform(0.8, 1.1), 2)
            urgency = random.choice(["low", "medium", "high"])
            source = random.choice(["Facebook", "Website", "Referral", "Instagram"])
            assigned_agent = None
            engagement_score = round(random.uniform(0.2, 1.0), 2)
            price_gap = round(budget - price, 2)
            conversion_probability = round(
                (engagement_score * 0.6) +
                (0.3 if urgency == "high" else 0.2 if urgency == "medium" else 0.1) +
                (0.1 if price_gap >= 0 else 0.05),
                2
            )
            conversion_status = 1 if conversion_probability > 0.75 else 0

            cursor.execute("""
                INSERT INTO leads 
                (property_id, budget, urgency_level, source, assigned_agent,
                 engagement_score, price_gap, conversion_probability, conversion_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                property_id, budget, urgency, source,
                assigned_agent, engagement_score,
                price_gap, conversion_probability,
                conversion_status
            ))

            leads_created += 1

    conn.commit()
    print("Leads created:", leads_created)
    print("DONE SUCCESSFULLY")

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("❌ خطأ في اسم المستخدم أو كلمة المرور")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("❌ قاعدة البيانات غير موجودة")
    else:
        print(f"❌ خطأ في قاعدة البيانات: {err}")
except Exception as e:
    print(f"❌ خطأ عام: {e}")
finally:
    if cursor:
        cursor.close()
        print("تم إغلاق cursor")
    if conn and conn.is_connected():
        conn.close()
        print("تم إغلاق الاتصال")