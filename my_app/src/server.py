# Скрипт flask сервера для предикта
import pickle

import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('hw1.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)


@app.route('/hello')
def hello_func():
    name = request.args.get('name')
    return f'hello, {name}', 200


@app.route('/predict', methods=['POST'])
def predict():
    print(request.json)
    numbers = np.array(list(request.json)).reshape(1, -1)
    print(numbers)
    result = model.predict(numbers)[0]
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
