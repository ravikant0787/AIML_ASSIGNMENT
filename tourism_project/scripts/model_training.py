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
import shutil
import tempfile

# Define the path to your dataset
DATA_FILE_PATH = 'tourism_project/data/tourism.csv'

def train_model():
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Dataset file not found at {DATA_FILE_PATH}. Ensure the data-prep job ran successfully.")
        exit(1)

    print(f"Loading data for model training from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH)

    CURRENT_WORKING_DIR = os.getcwd()

    # The temporary directory logic is now primarily handled by environment variables
    # set in pipeline.yml (TMPDIR, TEMP, MLFLOW_TMP_DIR).
    # Python's tempfile module should respect these.
    # Removed explicit 'tempfile.tempdir = ...' assignment to avoid potential conflicts
    # and rely on the shell environment variables.
    print(f"Python's default tempfile.tempdir: {tempfile.gettempdir()} (after system env vars applied)")

    # Debugging: Print current working directory and relevant environment variables
    print(f"Current working directory (inside script): {CURRENT_WORKING_DIR}")
    print(f"MLFLOW_TRACKING_URI env var (before explicit set): {os.environ.get('MLFLOW_TRACKING_URI')}")
    print(f"TMPDIR env var (from GitHub Actions): {os.environ.get('TMPDIR')}")
    print(f"TEMP env var (from GitHub Actions): {os.environ.get('TEMP')}")
    print(f"MLFLOW_TMP_DIR env var (from GitHub Actions): {os.environ.get('MLFLOW_TMP_DIR')}")


    # Explicitly define the local MLflow tracking URI path relative to the current working directory
    mlruns_local_path = os.path.join(CURRENT_WORKING_DIR, "tourism_project", "mlruns")

    # Define a specific artifact storage location for this run, separate from the general mlruns.
    # This path will be explicitly passed to mlflow.start_run's artifact_uri parameter.
    # It must also be an absolute path and preferably within the GitHub workspace.
    run_specific_artifact_location = os.path.join(CURRENT_WORKING_DIR, "tourism_project", "mlflow_run_artifacts")
    os.makedirs(run_specific_artifact_location, exist_ok=True)
    print(f"Run-specific artifact location created/ensured at: {run_specific_artifact_location}")


    # Clean up previous mlruns data for a fresh run
    if os.path.exists(mlruns_local_path):
        print(f"Cleaning up existing MLflow tracking directory: {mlruns_local_path}")
        shutil.rmtree(mlruns_local_path)

    os.makedirs(mlruns_local_path, exist_ok=True) # Ensure this local path is created

    print(f"MLflow will attempt to create/use tracking URI at: {mlruns_local_path}")

    # Set environment variable to allow MLflow filesystem tracking
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    # Also set MLFLOW_TRACKING_URI in environment for consistency with any subprocesses
    os.environ['MLFLOW_TRACKING_URI'] = f"file:{mlruns_local_path}"

    # Set MLflow tracking URI to the explicitly defined local directory
    mlflow.set_tracking_uri(f"file:{mlruns_local_path}")
    print(f"MLflow tracking URI explicitly set to: {mlflow.get_tracking_uri()}")

    # Set the experiment. The artifact_location for the experiment will be derived
    # from the tracking URI if not explicitly set here.
    # We will override the run's artifact_uri below directly in start_run.
    mlflow.set_experiment("tourism_package_prediction")
    print(f"MLflow experiment '{mlflow.get_experiment_by_name('tourism_package_prediction').name}' set.")


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
    # Explicitly set the artifact_uri for this run
    # This should be the most robust way to control where artifacts are saved for a run.
    with mlflow.start_run(artifact_uri=f"file://{run_specific_artifact_location}") as run:
        print(f"MLflow run started. Run ID: {run.info.run_id}")
        print(f"MLflow run artifact URI: {run.info.artifact_uri}") # Debug print

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
        # The model will be saved under `{run.info.artifact_uri}/random_forest_model`
        mlflow.sklearn.log_model(model_pipeline, "random_forest_model", registered_model_name="TourismPackagePredictor", serialization_format='cloudpickle')

    print("Model training, evaluation, and logging to MLflow complete. A model 'TourismPackagePredictor' has been registered.")

if __name__ == '__main__':
    train_model()
