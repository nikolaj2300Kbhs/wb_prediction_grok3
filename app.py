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
XAI_API_KEY = os.getenv('XAI_API_KEY')
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'  # Hypothetical endpoint; update as needed

def predict_box_cac(historical_data, future_box_info):
    """Predict the Customer Acquisition Cost (CAC) in euros for a future welcome box using Grok 3 API."""
    try:
        prompt = f"""
You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members at low Customer Acquisition Cost (CAC). Based on the historical data provided, which includes box features and their corresponding CAC in euros, predict the CAC for the future welcome box. The CAC should be a numerical value in euros, with two decimal places (e.g., 10.50). Consider factors such as the number of products, total retail value, number of unique categories, number of full-size products, number of premium products (>€20), total weight, average product rating, average brand rating, and average category rating. Return only the numerical CAC value in euros (e.g., 10.50).

Historical Data: {historical_data}

Future Box Info: {future_box_info}
"""
        headers = {
            'Authorization': f'Bearer {XAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        cacs = []
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
            logger.info(f"Sending request to xAI API: {payload}")
            response = requests.post(GROK_API_URL, json=payload, headers=headers)
            if response.status_code != 200:
                logger.error(f"xAI API error: {response.status_code} - {response.text}")
                raise Exception(f"xAI API error: {response.status_code} - {response.text}")
            result = response.json()
            cac = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            logger.info(f"Run response: {cac}")
            if not cac:
                logger.error("Model returned an empty response")
                raise ValueError("Empty response from model")
            try:
                cac_float = float(cac)
                if cac_float < 0:  # CAC should not be negative
                    raise ValueError("CAC cannot be negative")
                cacs.append(cac_float)
            except ValueError as e:
                logger.error(f"Invalid CAC format: {cac}, error: {str(e)}")
                raise ValueError(f"Invalid CAC: {cac}")
        if not cacs:
            raise ValueError("No valid CAC values collected")
        avg_cac = sum(cacs) / len(cacs)
        final_cac = f"{avg_cac:.2f}"
        logger.info(f"Averaged CAC from 5 runs: {final_cac}")
        return final_cac
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise Exception(f"Prediction error: {str(e)}")

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """Endpoint for predicting future box CAC in euros."""
    try:
        data = request.get_json()
        if not data or 'future_box_info' not in data:
            logger.error("Missing future box info")
            return jsonify({'error': 'Missing future_box_info'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        cac = predict_box_cac(historical_data, future_box_info)
        return jsonify({'predicted_cac': cac})  # Changed to return 'predicted_cac'
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
