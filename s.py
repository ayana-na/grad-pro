import requests
import numpy as np
import random

BASE = "http://localhost:5000"

def query(x1, x2):
    r = requests.get(f"{BASE}/query", params={"x1": x1, "x2": x2})
    return r.json()['output']

print("Collecting sample points...")
# جمع مجموعة نقاط عشوائية ومخرجاتها
num_samples = 200
X_samples = np.random.uniform(-5, 5, (num_samples, 2))
y_samples = np.array([query(x1, x2) for x1, x2 in X_samples])

def model_predict(w1, b1, w2, b2, X):
    # X shape: (N,2)
    h = np.maximum(0, X @ w1.T + b1)  # N x 3
    return h @ w2 + b2

def random_weights():
    w1 = np.random.randint(-3, 4, size=(3, 2))
    b1 = np.random.randint(-3, 4, size=3)
    w2 = np.random.randint(-3, 4, size=3)
    b2 = np.random.randint(-3, 4)
    return w1, b1, w2, b2

def evaluate(w1, b1, w2, b2):
    preds = model_predict(w1, b1, w2, b2, X_samples)
    return np.sum(np.abs(preds - y_samples) < 1e-6)  # عدد التطابقات

print("Searching for weights (this may take a few seconds)...")
best_score = 0
best_weights = None
trials = 0
while best_score < num_samples:
    trials += 1
    w1, b1, w2, b2 = random_weights()
    score = evaluate(w1, b1, w2, b2)
    if score > best_score:
        best_score = score
        best_weights = (w1, b1, w2, b2)
        print(f"  improved: {score}/{num_samples} matches")
    # إذا وجدنا تطابقاً كاملاً نخرج
    if best_score == num_samples:
        break

print(f"Found perfect match after {trials} trials.")
w1, b1, w2, b2 = best_weights
print("Weights:")
print("W1:\n", w1)
print("b1:", b1)
print("W2:", w2)
print("b2:", b2)

# تحضير الأوزان للتحقق
weights_list = w1.flatten().tolist() + b1.tolist() + w2.tolist() + [b2]
weights_list_int = [int(v) for v in weights_list]
print("Verifying on server...")
r = requests.post(f"{BASE}/verify", json={"weights": weights_list_int})
print(r.json())