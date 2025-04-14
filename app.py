from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.exposition import CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
request_count = Counter('goodiebox_api_requests_total', 'Total API requests', ['endpoint'])
error_count = Counter('goodiebox_api_errors_total', 'Total API errors', ['endpoint'])
success_count = Counter('goodiebox_api_successes_total', 'Total successful predictions', ['endpoint'])
response_time = Histogram('goodiebox_api_response_time_seconds', 'API response time', ['endpoint'])
grok_calls = Counter('goodiebox_grok_api_calls_total', 'Total Grok 3 API calls')
cac_distribution = Gauge('goodiebox_predicted_cac', 'Predicted CAC')
status_codes = Counter('goodiebox_api_status_codes_total', 'HTTP status codes returned', ['status_code'])

# Grok 3 API configuration
XAI_API_KEY = os.getenv('XAI_API_KEY')
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'  # Update with actual endpoint

def predict_box_cac(historical_data, future_box_info):
    """Predict the Customer Acquisition Cost (CAC) in euros for a future welcome box using Grok 3 API."""
    with response_time.labels(endpoint='/predict_box_score').time():
        try:
            prompt = f"""
You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members at low Customer Acquisition Cost (CAC). Based on the historical data provided, which includes box features and their corresponding CAC in euros, predict the CAC for the future welcome box. The CAC should be a numerical value in euros, with two decimal places (e.g., 10.50). Consider factors such as the number of products, total retail value, number of unique categories, number of full-size products, number of premium products (>€20), total weight, average product rating, average brand rating, average category rating, and niche products. Return only the numerical CAC value in euros (e.g., 10.50).

Historical Data: {historical_data}

Future Box Info: {future_box_info}
"""
            headers = {
                'Authorization': f'Bearer {XAI_API_KEY}',
                'Content-Type': 'application/json'
            }
            cacs = []
            for _ in range(10):  # Increased to 10 iterations
                payload = {
                    'model': 'grok-3',
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
                    'seed': 42
                }
                logger.info(f"Sending request to xAI API: {payload}")
                grok_calls.inc()
                response = requests.post(GROK_API_URL, json=payload, headers=headers)
                status_codes.labels(status_code=str(response.status_code)).inc()
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
                    if cac_float < 0:
                        raise ValueError("CAC cannot be negative")
                    cacs.append(cac_float)
                except ValueError as e:
                    logger.error(f"Invalid CAC format: {cac}, error: {str(e)}")
                    raise ValueError(f"Invalid CAC: {cac}")
            if not cacs:
                raise ValueError("No valid CAC values collected")
            avg_cac = sum(cacs) / len(cacs)
            final_cac = f"{avg_cac:.2f}"
            cac_distribution.set(float(final_cac))
            logger.info(f"Averaged CAC from 10 runs: {final_cac}")
            success_count.labels(endpoint='/predict_box_score').inc()
            return final_cac
        except Exception as e:
            error_count.labels(endpoint='/predict_box_score').inc()
            logger.error(f"Error in prediction: {str(e)}")
            raise Exception(f"Prediction error: {str(e)}")

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """Endpoint for predicting future box CAC in euros."""
    request_count.labels(endpoint='/predict_box_score').inc()
    try:
        data = request.get_json()
        if not data or 'future_box_info' not in data:
            logger.error("Missing future box info")
            error_count.labels(endpoint='/predict_box_score').inc()
            status_codes.labels(status_code='400').inc()
            return jsonify({'error': 'Missing future_box_info'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        cac = predict_box_cac(historical_data, future_box_info)
        status_codes.labels(status_code='200').inc()
        return jsonify({'predicted_cac': cac})
    except Exception as e:
        error_count.labels(endpoint='/predict_box_score').inc()
        status_codes.labels(status_code='500').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint to expose Prometheus metrics."""
    request_count.labels(endpoint='/metrics').inc()
    status_codes.labels(status_code='200').inc()
    return generate_latest(REGISTRY), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    request_count.labels(endpoint='/health').inc()
    status_codes.labels(status_code='200').inc()
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
