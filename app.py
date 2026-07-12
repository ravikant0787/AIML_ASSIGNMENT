import os
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify

# Set the MLflow tracking URI (ensure it points to where your model was registered)
# In a real deployment, this might be an MLflow server URI.
# For this example, we assume the model is accessible locally or from a configured MLflow server.
# If running locally from the 'tourism_project/mlruns' directory, adjust this.
# For simplicity, we'll assume the model is loaded by name from a local MLflow tracking server.
# You might need to adjust the tracking URI based on your MLflow setup in deployment.
# mlflow.set_tracking_uri("file:./tourism_project/mlruns") # Example if mlruns folder is packaged with the image

# Load the MLflow model using its registered name and latest version
# This assumes the model 'TourismPackagePredictor' was successfully registered
# and that the MLflow tracking server (even local) is accessible.
model_name = "TourismPackagePredictor"
model_version = "latest"
try:
    # Attempt to load the model from the local MLflow registry
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    print(f"Successfully loaded model: {model_name} version: {model_version}")
except Exception as e:
    print(f"Error loading MLflow model {model_name}/{model_version}: {e}")
    print("Make sure your MLflow tracking server is running and the model is registered.")
    print("For local testing, you might need to ensure the 'mlruns' directory is correctly placed or accessible.")
    loaded_model = None # Set to None to handle cases where model loading fails

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        json_ = request.json
        # Convert input JSON to a pandas DataFrame
        # It's crucial that the column names and order match the training data
        query_df = pd.DataFrame(json_)

        # Make predictions
        predictions = loaded_model.predict(query_df)

        # Return predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': loaded_model is not None}), 200

if __name__ == '__main__':
    # Use Gunicorn for production, Flask's built-in server for development
    # In the Dockerfile, Gunicorn will be used, so this block is mainly for local testing.
    app.run(host='0.0.0.0', port=5000, debug=True)
