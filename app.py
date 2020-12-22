import flask
from flask import Flask, jsonify, request
import json
import pickle
import numpy as np

app = Flask(__name__)

def load_models():
    file_name = "models/random_forest_model.pkl"
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model

@app.route('/predict', methods=['GET'])
def predict():
    # # parse input features from request
    request_json = request.get_json()
    x = request_json['input'].split(",")

    # input : type, bedrooms, bathrooms, square feet
    # x = ["apartment", 2, 2, 1000]

    transform_type = {"apartment":0, "condo":1, "house":2, "townhouse":3}
    x[0] = transform_type[x[0]]
    x = list(map(int, x))
    x = np.array([x])
    print(x)

    # load model
    model = load_models()
    prediction = model.predict(x)[0]
    response = json.dumps({'response': prediction})
    return response, 200
if __name__ == '__main__':
    application.run(debug=True)

# curl -X GET http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"input\":\"house,2,2,1000\"}"