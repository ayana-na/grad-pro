from flask import Flask, request, jsonify
import numpy as np
import random
import sys

app = Flask(__name__)

# توليد أوزان وانحيازات صحيحة عشوائية (2->3->1)
def generate_random_weights():
    w1 = np.array([[random.randint(-3, 3), random.randint(-3, 3)] for _ in range(3)])  # 3x2
    b1 = np.array([random.randint(-3, 3) for _ in range(3)])                          # 3
    w2 = np.array([random.randint(-3, 3) for _ in range(3)])                          # 1x3 -> 3
    b2 = random.randint(-3, 3)                                                         # scalar
    return w1, b1, w2, b2

# تخزين الأوزان الحقيقية (تتغير كل إعادة تشغيل)
true_weights = generate_random_weights()
w1, b1, w2, b2 = true_weights

def model_forward(x1, x2):
    x = np.array([x1, x2])
    h = np.maximum(0, w1 @ x + b1)   # ReLU
    y = np.dot(w2, h) + b2
    return float(y)

@app.route('/query')
def query():
    try:
        x1 = float(request.args['x1'])
        x2 = float(request.args['x2'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Missing or invalid x1, x2'}), 400
    y = model_forward(x1, x2)
    return jsonify({'output': y})

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    if not data or 'weights' not in data:
        return jsonify({'error': 'Send JSON with "weights" list of 13 numbers'}), 400
    submitted = data['weights']
    if len(submitted) != 13:
        return jsonify({'error': 'Need exactly 13 weights in order: w1_flat (6), b1 (3), w2 (3), b2 (1)'}), 400

    w1_flat = w1.flatten().tolist()  # 6
    b1_list = b1.tolist()            # 3
    w2_list = w2.tolist()            # 3
    b2_val  = b2                     # 1

    true_list = w1_flat + b1_list + w2_list + [b2_val]

    try:
        submitted_int = [int(round(v)) for v in submitted]
    except (ValueError, TypeError):
        return jsonify({'error': 'Weights must be numbers'}), 400

    if submitted_int == true_list:
        return jsonify({'flag': 'HTB{th1s_1s_4_dummy_fl4g}'})
    else:
        return jsonify({'result': 'Incorrect weights'})

if __name__ == '__main__':
    print("=" * 50)
    print("Server starting on http://0.0.0.0:5000")
    print("Press CTRL+C to stop")
    print("=" * 50)
    sys.stdout.flush()  # يضمن الطباعة الفورية
    app.run(host='0.0.0.0', port=5000, debug=False)