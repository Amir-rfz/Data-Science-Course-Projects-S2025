import os
import subprocess
import sys
import argparse 
import pandas as pd
import sqlite3
from load_data import load_data_from_db
from feature_engineering import engineer_features
from preprocess import preprocess_data 
from train_model import train_model 
from predict_on_saved_test import predict_and_evaluate 
import mlflow
import datetime 
from load_data import load_data_from_db
from feature_engineering import engineer_features
from preprocess import preprocess_data
from train_model import train_model
from predict_on_saved_test import predict_and_evaluate

def run_script(script_name, stage_name):
    try:
        print(f"\n----- Running Stage: {stage_name} ({script_name}) -----")
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"--- Stderr from {script_name} ---")
            print(result.stderr)
        print(f"----- Stage: {stage_name} Completed -----")
        return True
    except subprocess.CalledProcessError as e:
        print(f"----- Error in Stage: {stage_name} ({script_name}) -----")
        return False
    except FileNotFoundError:
        print(f"----- Error: Script {script_name} not found. -----")
        return False

def main():
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Pipeline with MLflow")
    parser.add_argument("action", choices=["train", "prediction"])
    parser.add_argument("--experiment_name", type=str, default="Flight_Delay_Prediction")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    database_dir = os.path.join(project_root, "Database")
    models_dir = os.path.join(project_root, "Models") 

    db_path = os.path.join(database_dir, "flight_data.db")

    os.makedirs(database_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True) 
    mlflow.set_experiment(args.experiment_name)

    if args.action == "train":
        with mlflow.start_run(run_name=f"Training_Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID for Training: {run_id}")
            mlflow.log_param("pipeline_action", "train")

            print("Starting 'train' workflow with MLflow logging...")
            create_db_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_database.py") 
            if not run_script(create_db_script, "Database Creation"): return

            print("\n----- Running Stage: Data Loading, Feature Engineering, Preprocessing, Model Training, and Saving Test Data -----")
            print("Step 1: Loading data...")
            flights_raw_df, airports_raw_df, airlines_raw_df = load_data_from_db(db_path)

            print("Step 2: Engineering features...")
            engineered_df = engineer_features(flights_raw_df, airports_raw_df, airlines_raw_df)

            print("Step 3: Preprocessing data (fitting preprocessor)...")
            preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")
            X_processed_full, y_target_full, flight_ids_full = preprocess_data(
                engineered_df.copy(),
                preprocessor_path=preprocessor_save_path,
                fit_preprocessor=True
            )
            if os.path.exists(preprocessor_save_path):
                mlflow.log_artifact(preprocessor_save_path, artifact_path="preprocessor")
                print(f"Logged {preprocessor_save_path} to MLflow artifacts.")


            print("Step 4: Training model and saving test set to DB...")
            trained_model_path = train_model(X_processed_full, y_target_full, flight_ids_full, models_dir, db_path)
            
            if trained_model_path:
                mlflow.log_param("training_status", "completed")
                if os.path.exists(db_path):
                    print(f"Logging database '{db_path}' as an MLflow artifact...")
                    mlflow.log_artifact(local_path=db_path, artifact_path="database_after_train")
                    print(f"Database '{db_path}' logged successfully to MLflow artifacts under 'database_snapshot'.")
                else:
                    print(f"WARNING: Database file '{db_path}' not found. Cannot log it as an artifact.")
                print(f"----- 'train' workflow completed successfully. Model available at {trained_model_path} and logged to MLflow. -----")
            else:
                mlflow.log_param("training_status", "failed_model_training")
                print("----- 'train' workflow failed during model training. -----")



    elif args.action == "prediction":
        with mlflow.start_run(run_name=f"Prediction_Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID for Prediction: {run_id}")
            mlflow.log_param("pipeline_action", "prediction")

            print("Starting 'prediction' workflow with MLflow logging...")
            print("Running prediction and evaluation on saved test data...")
            predict_and_evaluate(models_dir, db_path)

            mlflow.log_param("prediction_status", "completed")

            if os.path.exists(db_path):
                print(f"Logging database '{db_path}' as an MLflow artifact...")
                mlflow.log_artifact(local_path=db_path, artifact_path="database_after_prediction")
                print(f"Database '{db_path}' logged successfully to MLflow artifacts under 'database_snapshot'.")
            else:
                print(f"WARNING: Database file '{db_path}' not found. Cannot log it as an artifact.")

            print("----- 'prediction' workflow completed. -----")

if __name__ == "__main__":
    main()