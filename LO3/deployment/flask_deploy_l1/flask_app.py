from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --------------------------
# Load trained model and scaler
# --------------------------
try:
    lasso_model = joblib.load("tuning/model/lasso_model.pkl")
    scaler = joblib.load("tuning/model/scaler_lasso.pkl")
    print("✅ Model and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
    lasso_model = None
    scaler = None

# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for house price prediction"""
    try:
        if lasso_model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist in the model/ directory.'
            }), 500
        
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['size_sqft', 'num_bedrooms', 'num_bathrooms', 'age_years', 'location_score']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input for prediction
        X_input = pd.DataFrame([[
            float(data['size_sqft']),
            int(data['num_bedrooms']),
            int(data['num_bathrooms']),
            float(data['age_years']),
            float(data['location_score'])
        ]], columns=required_fields)
        
        # Scale the input
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        price_pred = lasso_model.predict(X_scaled)[0]
        
        return jsonify({
            'predicted_price': round(float(price_pred), 2),
            'input_features': data,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """API endpoint to get model coefficients and info"""
    try:
        if lasso_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        feature_names = ["size_sqft", "num_bedrooms", "num_bathrooms", "age_years", "location_score"]
        coefficients = dict(zip(feature_names, lasso_model.coef_.tolist()))
        
        return jsonify({
            'coefficients': coefficients,
            'intercept': float(lasso_model.intercept_),
            'feature_names': feature_names
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if lasso_model is not None and scaler is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

# --------------------------
# Error handlers
# --------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)