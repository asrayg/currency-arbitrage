from flask import Flask, request, jsonify
import pandas as pd
from feature_extraction import extract_features
from model_training import model
from recommendation_system import generate_recommendations

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    content = request.json
    text_data = content['texts']
    features = extract_features(text_data)
    predictions = model.predict(features)
    recommendations = generate_recommendations(predictions)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
