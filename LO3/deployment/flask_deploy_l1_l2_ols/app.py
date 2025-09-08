from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# ------------------------------
# Load scaler and model parameters
# ------------------------------
try:
    scaler = joblib.load("tuning/models_l1_l2_ols/scaler.pkl")
    
    # OLS
    ols_params = np.load("tuning/models_l1_l2_ols/ols_model_params.npz")
    ols_coef = ols_params["coef"]
    ols_intercept = ols_params["intercept"]
    
    # Ridge
    ridge_params = np.load("tuning/models_l1_l2_ols/ridge_model_params.npz")
    ridge_coef = ridge_params["coef"]
    ridge_intercept = ridge_params["intercept"]
    
    # Lasso
    lasso_params = np.load("tuning/models_l1_l2_ols/lasso_model_params.npz")
    lasso_coef = lasso_params["coef"]
    lasso_intercept = lasso_params["intercept"]
    
    print("✅ All models and scaler loaded successfully!")
    models_loaded = True
    
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
    models_loaded = False
    scaler = None
    ols_coef = ols_intercept = None
    ridge_coef = ridge_intercept = None
    lasso_coef = lasso_intercept = None

# ------------------------------
# Prediction function
# ------------------------------
def predict(model_name, X_scaled):
    """Make prediction using specified model"""
    if model_name == "OLS":
        return np.dot(X_scaled, ols_coef) + ols_intercept
    elif model_name == "Ridge":
        return np.dot(X_scaled, ridge_coef) + ridge_intercept
    elif model_name == "Lasso":
        return np.dot(X_scaled, lasso_coef) + lasso_intercept
    else:
        return None

def get_model_coefficients(model_name):
    """Get coefficients for specified model"""
    feature_names = ["size_sqft", "num_bedrooms", "num_bathrooms", "age_years", "location_score"]
    
    if model_name == "OLS":
        return dict(zip(feature_names, ols_coef.tolist())), float(ols_intercept)
    elif model_name == "Ridge":
        return dict(zip(feature_names, ridge_coef.tolist())), float(ridge_intercept)
    elif model_name == "Lasso":
        return dict(zip(feature_names, lasso_coef.tolist())), float(lasso_intercept)
    else:
        return None, None

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for house price prediction"""
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please ensure model files exist in the models_l1_l2_ols/ directory.'
            }), 500
        
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['size_sqft', 'num_bedrooms', 'num_bathrooms', 'age_years', 'location_score', 'model_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate model name
        model_name = data['model_name']
        if model_name not in ['OLS', 'Ridge', 'Lasso']:
            return jsonify({'error': 'Invalid model name. Must be OLS, Ridge, or Lasso.'}), 400
        
        # Prepare input for prediction
        X_input = np.array([[
            float(data['size_sqft']),
            int(data['num_bedrooms']),
            int(data['num_bathrooms']),
            float(data['age_years']),
            float(data['location_score'])
        ]])
        
        # Scale the input
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        prediction = predict(model_name, X_scaled)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'predicted_price': round(float(prediction[0]), 2),
            'model_used': model_name,
            'input_features': {k: v for k, v in data.items() if k != 'model_name'},
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/model-info/<model_name>', methods=['GET'])
def model_info(model_name):
    """API endpoint to get model coefficients and info"""
    try:
        if not models_loaded:
            return jsonify({'error': 'Models not loaded'}), 500
        
        if model_name not in ['OLS', 'Ridge', 'Lasso']:
            return jsonify({'error': 'Invalid model name. Must be OLS, Ridge, or Lasso.'}), 400
        
        coefficients, intercept = get_model_coefficients(model_name)
        
        if coefficients is None:
            return jsonify({'error': 'Failed to get model coefficients'}), 500
        
        return jsonify({
            'model_name': model_name,
            'coefficients': coefficients,
            'intercept': intercept,
            'feature_names': list(coefficients.keys())
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def available_models():
    """API endpoint to get available models"""
    return jsonify({
        'available_models': ['OLS', 'Ridge', 'Lasso'],
        'models_loaded': models_loaded,
        'model_descriptions': {
            'OLS': 'Ordinary Least Squares - Linear regression without regularization',
            'Ridge': 'Ridge Regression - L2 regularization to prevent overfitting',
            'Lasso': 'Lasso Regression - L1 regularization with feature selection'
        }
    })

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """API endpoint to compare predictions from all models"""
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please ensure model files exist in the models_l1_l2_ols/ directory.'
            }), 500
        
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['size_sqft', 'num_bedrooms', 'num_bathrooms', 'age_years', 'location_score']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input for prediction
        X_input = np.array([[
            float(data['size_sqft']),
            int(data['num_bedrooms']),
            int(data['num_bathrooms']),
            float(data['age_years']),
            float(data['location_score'])
        ]])
        
        # Scale the input
        X_scaled = scaler.transform(X_input)
        
        # Make predictions with all models
        predictions = {}
        for model_name in ['OLS', 'Ridge', 'Lasso']:
            prediction = predict(model_name, X_scaled)
            predictions[model_name] = round(float(prediction[0]), 2)
        
        # Calculate statistics
        prices = list(predictions.values())
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        return jsonify({
            'predictions': predictions,
            'statistics': {
                'average': round(avg_price, 2),
                'minimum': min_price,
                'maximum': max_price,
                'range': round(price_range, 2)
            },
            'input_features': data,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Comparison failed: {str(e)}',
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'available_models': ['OLS', 'Ridge', 'Lasso'] if models_loaded else []
    })

# ------------------------------
# Error handlers
# ------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)