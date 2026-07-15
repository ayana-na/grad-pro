import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

print("Connecting to database...")

try:
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="real_estate_ai",
        use_pure=True,
        connection_timeout=10
    )
    print("Connected successfully!")
    
    cursor = conn.cursor()
    query = """
        SELECT budget, urgency_level, source, engagement_score, price_gap, conversion_status
        FROM leads
    """
    cursor.execute(query)
    print("Query executed.")
    
    rows = cursor.fetchall()
    print(f"Fetched {len(rows)} rows.")
    
    columns = ["budget", "urgency_level", "source", "engagement_score", "price_gap", "conversion_status"]
    df = pd.DataFrame(rows, columns=columns)
    print("DataFrame created.")
    
    cursor.close()
    conn.close()
    print("Connection closed.")

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("❌ خطأ في اسم المستخدم أو كلمة المرور")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("❌ قاعدة البيانات غير موجودة")
    else:
        print(f"❌ خطأ MySQL: {err}")
    exit()
except Exception as e:
    print(f"❌ خطأ عام: {e}")
    exit()

print("Dataset size:", len(df))

# ... باقي الكود


# -------------------------
# تجهيز البيانات
# -------------------------

df = pd.get_dummies(df, columns=["urgency_level", "source"])

X = df.drop("conversion_status", axis=1)
y = df["conversion_status"]

print("Number of features:", len(X.columns))


# -------------------------
# Cross Validation
# -------------------------

print("\nRunning 5-Fold Stratified Cross Validation...")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracies = []

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    acc = (preds == y_test).mean()
    accuracies.append(acc)

    print("\n-----------------------------")
    print(f"Fold {fold} Accuracy:", acc)
    print("-----------------------------")

    print(classification_report(y_test, preds))


print("\nAverage Accuracy:", np.mean(accuracies))


# -------------------------
# تدريب النموذج النهائي
# -------------------------

print("\nTraining final model on full dataset...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)


# -------------------------
# Feature Importance
# -------------------------

importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)


# -------------------------
# حفظ النموذج
# -------------------------

joblib.dump(model, "lead_model.pkl")
joblib.dump(X.columns, "lead_model_columns.pkl")

print("\nModel saved successfully!")