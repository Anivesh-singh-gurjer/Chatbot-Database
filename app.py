from flask import Flask, request, jsonify
import joblib
import re
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model and TF-IDF vectorizer
model = joblib.load("symptom_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# LINE API details
LINE_ACCESS_TOKEN = 'FvNIn8DpgSPrOVgJw8XBVyfebywKa1zrOBg2slg80u/i5GemHCylgslLsNQH9m50vcof0NPPqR4YGeX/UT/hpW5kzFtMcK86doX7H7FzIc1zdfm4ayXJf80WW17h77fAxU2tXEkBRDX7a+3uleJ4sQdB04t89/1O/w1cDnyilFU='
LINE_API_URL = 'https://api.line.me/v2/bot/message/reply'

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    # Get the symptom description from the request
    data = request.get_json()
    symptoms = data['symptoms']  # Get the symptoms text

    # Preprocess and vectorize the symptoms
    symptoms_cleaned = preprocess_text(symptoms)  # Apply your text preprocessing
    symptoms_vectorized = tfidf.transform([symptoms_cleaned])

    # Make a prediction with probabilities
    prediction_probs = model.predict_proba(symptoms_vectorized)[0]
    top_indices = prediction_probs.argsort()[-3:][::-1]  # Get top 3 diseases

    # Check if the top probabilities are close (within a threshold, e.g., 10%)
    prob_diff_threshold = 0.10
    if np.max(prediction_probs) - np.min(prediction_probs[top_indices]) < prob_diff_threshold:
        response_message = f"The probabilities are close. More information is needed."
    else:
        response_message = ""

    # Construct the response message with top 3 diseases and their probabilities
    top_3_diseases = [(model.classes_[i], prediction_probs[i]) for i in top_indices]
    response_message += "The top 3 predicted diseases are:\n"
    for disease, prob in top_3_diseases:
        response_message += f"{disease}: {prob*100:.2f}%\n"

    return jsonify({'message': response_message})

@app.route('/webhook', methods=['POST'])
def webhook():
    body = request.get_json()

    # Extract message and reply token from LINE webhook
    events = body.get('events', [])
    for event in events:
        if event['type'] == 'message':
            reply_token = event['replyToken']
            user_message = event['message']['text']
            
            # Use your symptom model to predict the disease
            symptoms_cleaned = preprocess_text(user_message)
            symptoms_vectorized = tfidf.transform([symptoms_cleaned])
            prediction_probs = model.predict_proba(symptoms_vectorized)[0]
            top_indices = prediction_probs.argsort()[-3:][::-1]  # Get top 3 diseases
            
            # Check if the probabilities are close
            prob_diff_threshold = 0.10
            if np.max(prediction_probs) - np.min(prediction_probs[top_indices]) < prob_diff_threshold:
                reply_message = "The probabilities are close. More information is needed.\n"
            else:
                reply_message = ""
            
            # Construct the reply message with top 3 diseases and probabilities
            top_3_diseases = [(model.classes_[i], prediction_probs[i]) for i in top_indices]
            reply_message += "The top 3 predicted diseases are:\n"
            for disease, prob in top_3_diseases:
                reply_message += f"{disease}: {prob*100:.2f}%\n"
            
            # Send the reply back to LINE
            send_reply(reply_token, reply_message)

    return 'OK'

def send_reply(reply_token, message):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}',
    }

    body = {
        'replyToken': reply_token,
        'messages': [{
            'type': 'text',
            'text': message
        }]
    }

    requests.post(LINE_API_URL, headers=headers, data=json.dumps(body))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

