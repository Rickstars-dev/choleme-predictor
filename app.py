"""
Flask web application for cholesterol level prediction
REGRESSION model - predicts continuous cholesterol values (mg/dL)
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and metadata
# Use absolute paths for deployment compatibility
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cholesterol_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'feature_names.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'metadata.pkl')

try:
    print(f"🔍 Looking for models in: {BASE_DIR}")
    print(f"   Model path: {MODEL_PATH}")
    print(f"   Model exists: {os.path.exists(MODEL_PATH)}")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    metadata = joblib.load(METADATA_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: {metadata['model_name']}")
    print(f"   Predicts: {metadata['target']} ({metadata['unit']})")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files in current dir: {os.listdir('.')}")
    if os.path.exists('model'):
        print(f"   Files in model dir: {os.listdir('model')}")
    print("Please run 'python train_model.py' first to train the model.")
    model = None
    scaler = None
    feature_names = None
    metadata = None

@app.route('/')
def landing():
    """
    Render the landing page
    """
    return render_template('landing.html')

@app.route('/predict-page')
def predict_page():
    """
    Render the prediction form page
    """
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make cholesterol level prediction based on user input
    """
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get input data from form
        data = request.get_json()
        
        # Extract features in the correct order
        features = []
        for feature_name in feature_names:
            value = data.get(feature_name, 0)
            features.append(float(value))
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction (REGRESSION - predicts continuous value)
        predicted_cholesterol = model.predict(features_scaled)[0]
        
        # Round to 1 decimal place
        predicted_cholesterol = round(predicted_cholesterol, 1)
        
        # Determine cholesterol level category
        if predicted_cholesterol < 200:
            level = "Desirable"
            level_color = "green"
            message = "Your cholesterol level is in the healthy range!"
            recommendation = "Maintain your current healthy lifestyle."
        elif predicted_cholesterol < 240:
            level = "Borderline High"
            level_color = "orange"
            message = "Your cholesterol level is borderline high."
            recommendation = "Consider lifestyle changes: improve diet, increase exercise."
        else:
            level = "High"
            level_color = "red"
            message = "Your cholesterol level is high."
            recommendation = "Consult a healthcare provider. Lifestyle changes and medication may be needed."
        
        # Return prediction result
        return jsonify({
            'predicted_cholesterol': predicted_cholesterol,
            'unit': 'mg/dL',
            'level': level,
            'level_color': level_color,
            'message': message,
            'recommendation': recommendation,
            'model_type': 'Regression',
            'interpretation': {
                'desirable': '< 200 mg/dL',
                'borderline': '200-239 mg/dL',
                'high': '≥ 240 mg/dL'
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 400

@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': metadata['model_name'] if metadata else 'Unknown',
        'prediction_type': 'Regression (Continuous Values)'
    })

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
