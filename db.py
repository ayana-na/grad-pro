import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="real_estate_ai"
    )
    cursor = conn.cursor()
    print("✅ تم الاتصال بقاعدة البيانات بنجاح!")

    # اختبر الاتصال بعرض قواعد البيانات
    cursor.execute("SHOW DATABASES;")
    print("قواعد البيانات الموجودة:")
    for db in cursor:
        print(f" - {db[0]}")

    cursor.close()
    conn.close()
except mysql.connector.Error as err:
    print(f"❌ خطأ: {err}")
    
    CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role ENUM('admin','agent','manager','hr') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
    