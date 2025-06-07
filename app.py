from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from utils.run_model import predict_label


BASE_URL = '/api/v1'
app = Flask("Coffee Nutricionists API")
MODEL_PATH = 'runs/detect/train/weights/best.pt'

CORS(app, resources={r"/api/v1/*": {"origins": "*"}})

@app.route(BASE_URL)
def index():
    return jsonify(message="Welcome to the Coffee Nutricionists API!"), 200

@app.route(BASE_URL + '/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify(error="No image uploaded."), 400
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    result = predict_label(image, MODEL_PATH, conf=0.25)
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081)