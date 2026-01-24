"""
Flask API Server for Parkinson's Disease Prediction
Provides endpoints for all models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js integration

# Load all models and scalers
models = {}
scalers = {}
features = {}

def load_model(model_name):
    """Load a specific model with its scaler and features"""
    try:
        models[model_name] = joblib.load(f'models/{model_name}_model.pkl')
        scalers[model_name] = joblib.load(f'models/{model_name}_scaler.pkl')
        with open(f'models/{model_name}_features.json', 'r') as f:
            features[model_name] = json.load(f)
        return True
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        return False

# Load all models at startup
model_names = ['spiral_svm', 'tapping_rf', 'reaction_lr', 'voice_svm', 
               'nonmotor_nb', 'cognitive_gb', 'ensemble_final']

for model_name in model_names:
    load_model(model_name)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys())
    })

@app.route('/predict/spiral', methods=['POST'])
def predict_spiral():
    """Spiral drawing test prediction"""
    try:
        data = request.get_json()
        features_data = data.get('features', {})
        
        # Create dataframe with correct feature order
        df = pd.DataFrame([features_data])[features['spiral_svm']]
        
        # Scale and predict
        scaled_data = scalers['spiral_svm'].transform(df)
        prediction = models['spiral_svm'].predict(scaled_data)[0]
        probability = models['spiral_svm'].predict_proba(scaled_data)[0]
        
        return jsonify({
            'test': 'spiral',
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/tapping', methods=['POST'])
def predict_tapping():
    """Tapping test prediction"""
    try:
        data = request.get_json()
        features_data = data.get('features', {})
        
        df = pd.DataFrame([features_data])[features['tapping_rf']]
        scaled_data = scalers['tapping_rf'].transform(df)
        prediction = models['tapping_rf'].predict(scaled_data)[0]
        probability = models['tapping_rf'].predict_proba(scaled_data)[0]
        
        return jsonify({
            'test': 'tapping',
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/reaction', methods=['POST'])
def predict_reaction():
    """Reaction test prediction"""
    try:
        data = request.get_json()
        features_data = data.get('features', {})
        
        df = pd.DataFrame([features_data])[features['reaction_lr']]
        scaled_data = scalers['reaction_lr'].transform(df)
        prediction = models['reaction_lr'].predict(scaled_data)[0]
        probability = models['reaction_lr'].predict_proba(scaled_data)[0]
        
        return jsonify({
            'test': 'reaction',
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/voice', methods=['POST'])
def predict_voice():
    """Voice test prediction"""
    try:
        data = request.get_json()
        features_data = data.get('features', {})
        
        df = pd.DataFrame([features_data])[features['voice_svm']]
        scaled_data = scalers['voice_svm'].transform(df)
        prediction = models['voice_svm'].predict(scaled_data)[0]
        probability = models['voice_svm'].predict_proba(scaled_data)[0]
        
        return jsonify({
            'test': 'voice',
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/nonmotor', methods=['POST'])
def predict_nonmotor():
    """Non-motor test prediction"""
    try:
        data = request.get_json()
        features_data = data.get('features', {})
        
        df = pd.DataFrame([features_data])[features['nonmotor_nb']]
        scaled_data = scalers['nonmotor_nb'].transform(df)
        prediction = models['nonmotor_nb'].predict(scaled_data)[0]
        probability = models['nonmotor_nb'].predict_proba(scaled_data)[0]
        
        return jsonify({
            'test': 'nonmotor',
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/cognitive', methods=['POST'])
def predict_cognitive():
    """Cognitive test prediction"""
    try:
        data = request.get_json()
        features_data = data.get('features', {})
        
        df = pd.DataFrame([features_data])[features['cognitive_gb']]
        scaled_data = scalers['cognitive_gb'].transform(df)
        prediction = models['cognitive_gb'].predict(scaled_data)[0]
        probability = models['cognitive_gb'].predict_proba(scaled_data)[0]
        
        return jsonify({
            'test': 'cognitive',
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/ensemble', methods=['POST'])
def predict_ensemble():
    """Final risk assessment using ensemble model"""
    try:
        data = request.get_json()
        features_data = data.get('features', {})
        
        df = pd.DataFrame([features_data])[features['ensemble_final']]
        scaled_data = scalers['ensemble_final'].transform(df)
        prediction = models['ensemble_final'].predict(scaled_data)[0]
        probability = models['ensemble_final'].predict_proba(scaled_data)[0]
        
        return jsonify({
            'test': 'ensemble',
            'prediction': int(prediction),
            'probability': {
                'healthy': float(probability[0]),
                'parkinsons': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1]),
            'confidence': float(max(probability))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/comprehensive', methods=['POST'])
def predict_comprehensive():
    """Comprehensive prediction using all available test data"""
    try:
        data = request.get_json()
        results = {}
        
        # Run predictions for each test if data is provided
        test_mapping = {
            'spiral': 'spiral_svm',
            'tapping': 'tapping_rf',
            'reaction': 'reaction_lr',
            'voice': 'voice_svm',
            'nonmotor': 'nonmotor_nb',
            'cognitive': 'cognitive_gb'
        }
        
        parkinsons_probabilities = []
        
        for test_name, model_name in test_mapping.items():
            if test_name in data:
                try:
                    df = pd.DataFrame([data[test_name]])[features[model_name]]
                    scaled_data = scalers[model_name].transform(df)
                    prediction = models[model_name].predict(scaled_data)[0]
                    probability = models[model_name].predict_proba(scaled_data)[0]
                    
                    results[test_name] = {
                        'prediction': int(prediction),
                        'probability': float(probability[1])
                    }
                    parkinsons_probabilities.append(probability[1])
                except Exception as e:
                    results[test_name] = {'error': str(e)}
        
        # Calculate overall risk
        if parkinsons_probabilities:
            avg_probability = np.mean(parkinsons_probabilities)
            overall_prediction = 1 if avg_probability > 0.5 else 0
            
            return jsonify({
                'individual_tests': results,
                'overall_assessment': {
                    'prediction': int(overall_prediction),
                    'average_probability': float(avg_probability),
                    'risk_level': get_risk_level(avg_probability),
                    'tests_completed': len(parkinsons_probabilities)
                }
            })
        else:
            return jsonify({'error': 'No valid test data provided'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Moderate'
    elif probability < 0.8:
        return 'High'
    else:
        return 'Very High'

@app.route('/features/<model_name>', methods=['GET'])
def get_features(model_name):
    """Get required features for a specific model"""
    if model_name in features:
        return jsonify({
            'model': model_name,
            'features': features[model_name]
        })
    return jsonify({'error': 'Model not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)