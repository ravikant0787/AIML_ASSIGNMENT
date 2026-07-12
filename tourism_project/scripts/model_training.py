import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import numpy as np
import os

# Define the path to your dataset
DATA_FILE_PATH = 'tourism_project/data/tourism.csv'

def train_model():
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Dataset file not found at {DATA_FILE_PATH}. Ensure the data-prep job ran successfully.")
        exit(1)

    print(f"Loading data for model training from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH)

    # Create the directory for MLflow artifacts if it doesn't exist
    os.makedirs("tourism_project/mlruns", exist_ok=True)

    # Set environment variable to allow MLflow filesystem tracking
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

    # Set MLflow tracking URI to a local directory for now
    mlflow.set_tracking_uri("file:./tourism_project/mlruns")
    mlflow.set_experiment("tourism_package_prediction")

    # Define target and features
    target = 'ProdTaken'
    features_to_drop = ['Unnamed: 0', 'CustomerID', target]
    X = df.drop(columns=features_to_drop)
    y = df[target]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any other columns not explicitly transformed
    )

    # Create a full pipeline with preprocessing and a classifier
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=42))])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Starting MLflow run for model training...")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state_split", 42)
        mlflow.log_param("classifier_model", "RandomForestClassifier")
        classifier_params = model_pipeline.named_steps['classifier'].get_params()
        for param, value in classifier_params.items():
            mlflow.log_param(f"rf_{param}", value)

        # Train the model
        model_pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = model_pipeline.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Model Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Log and register the model with cloudpickle serialization
        mlflow.sklearn.log_model(model_pipeline, "random_forest_model", registered_model_name="TourismPackagePredictor", serialization_format='cloudpickle')

    print("Model training, evaluation, and logging to MLflow complete. A model 'TourismPackagePredictor' has been registered.")

if __name__ == '__main__':
    train_model()
