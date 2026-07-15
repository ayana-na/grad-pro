import random
import mysql.connector

# الاتصال بقاعدة البيانات
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="real_estate_ai"
)

cursor = conn.cursor()

for i in range(3000):

    budget = random.randint(50000, 500000)
    urgency = random.randint(1, 5)
    agent_experience = random.randint(1, 10)

    # منطق شبه واقعي للـ conversion
    conversion_probability = (
        (budget / 500000) * 0.4 +
        (urgency / 5) * 0.4 +
        (agent_experience / 10) * 0.2
    )

    converted = 1 if conversion_probability > 0.6 else 0

    cursor.execute("""
        INSERT INTO leads (budget, urgency, agent_experience, converted)
        VALUES (%s, %s, %s, %s)
    """, (budget, urgency, agent_experience, converted))

conn.commit()
cursor.close()
conn.close()

print("Data Inserted Successfully")