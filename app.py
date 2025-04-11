from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Grok 3 API configuration
GROK_API_KEY = os.getenv('GROK_API_KEY')
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'  # Hypothetical endpoint; update as needed

def predict_box_score(historical_data, future_box_info):
    """Predict an attractiveness score (1-5, two decimals) using Grok 3 API."""
    try:
        prompt = f"""
You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members at low Customer Acquisition Cost (CAC). Based on the historical data provided, which includes box features and their corresponding CAC, predict an attractiveness score for the future welcome box. The score should be on a scale of 1 to 5, with two decimal places (e.g., 4.23), where 5 indicates the box is very likely to have a low CAC, and 1 indicates it is likely to have a high CAC. Consider factors such as the number of products, total retail value, number of unique categories, number of full-size products, number of premium products (>€20), total weight, average product rating, average brand rating, and average category rating. Return only the numerical score (e.g., 4.23).

Historical Data: {historical_data}

Future Box Info: {future_box_info}
"""
        headers = {
            'Authorization': f'Bearer {GROK_API_KEY}',
            'Content-Type': 'application/json'
        }
        scores = []
        for _ in range(5):  # Run 5 times and average
            payload = {
                'model': 'grok-3',  # Hypothetical model name
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert in predicting Goodiebox performance, skilled at analyzing historical trends.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 10,
                'temperature': 0,
                'seed': 42  # For reproducibility, if supported
            }
            response = requests.post(GROK_API_URL, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for bad status codes
            result = response.json()
            score = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            logger.info(f"Run response: {score}")
            if not score:
                logger.error("Model returned an empty response")
                raise ValueError("Empty response from model")
            try:
                score_float = float(score)
                if not (1 <= score_float <= 5):
                    raise ValueError("Score out of range")
                scores.append(score_float)
            except ValueError as e:
                logger.error(f"Invalid score format: {score}, error: {str(e)}")
                raise ValueError(f"Invalid score: {score}")
        if not scores:
            raise ValueError("No valid scores collected")
        avg_score = sum(scores) / len(scores)
        final_score = f"{avg_score:.2f}"
        logger.info(f"Averaged score from 5 runs: {final_score}")
        return final_score
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise Exception(f"Prediction error: {str(e)}")

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """Endpoint for predicting future box attractiveness scores."""
    try:
        data = request.get_json()
        if not data or 'future_box_info' not in data:
            logger.error("Missing future box info")
            return jsonify({'error': 'Missing future_box_info'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        score = predict_box_score(historical_data, future_box_info)
        return jsonify({'predicted_box_score': score})
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not GROK_API_KEY:
        raise ValueError("GROK_API_KEY not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
