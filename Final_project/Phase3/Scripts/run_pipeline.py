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
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Pipeline")
    parser.add_argument("action", choices=["train", "prediction"])
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    database_dir = os.path.join(project_root, "Database")
    models_dir = os.path.join(project_root, "Models")
    scripts_dir = os.path.join(project_root, "Scripts") 
    
    db_path = os.path.join(database_dir, "flight_data.db")
    
    os.makedirs(database_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if args.action == "train":
        print("Starting 'train' workflow...")
        create_db_script = os.path.join(scripts_dir, "create_database.py")
        if not run_script(create_db_script, "Database Creation"): return

        print("\n----- Running Stage: Data Loading, Feature Engineering, Preprocessing, Model Training, and Saving Test Data -----")
        
        print("Step 1: Loading data...")
        flights_raw_df, airports_raw_df, airlines_raw_df = load_data_from_db(db_path)
        if flights_raw_df.empty:
            print("Failed to load data. Aborting training pipeline.")
            return

        print("Step 2: Engineering features...")
        engineered_df = engineer_features(flights_raw_df, airports_raw_df, airlines_raw_df)


        print("Step 3: Preprocessing data...")
        preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")
        X_processed_full, y_target_full, flight_ids_full = preprocess_data(
            engineered_df.copy(), 
            preprocessor_path=preprocessor_save_path, 
            fit_preprocessor=True
        )

        print("Step 4: Training model and saving test set to DB...")
        train_model(X_processed_full, y_target_full, flight_ids_full, models_dir, db_path) 

        print("----- 'train' workflow completed. -----")

    elif args.action == "prediction":
        print("Starting 'prediction' workflow...")
    
        print("Running prediction and evaluation on saved test data...")
        predict_and_evaluate(models_dir, db_path)
        
        print("----- 'prediction' workflow completed. -----")
            
if __name__ == "__main__":
    main()