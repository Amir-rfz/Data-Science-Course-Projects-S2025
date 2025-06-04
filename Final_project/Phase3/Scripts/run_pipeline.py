# # Scripts/run_pipeline.py
# import os
# import subprocess # To call other python scripts
# import sys

# # Ensure the Scripts directory is in the Python path if running from elsewhere
# # or if scripts import from each other using relative paths that assume Scripts is current dir.
# # However, it's better if imports are structured (e.g. from project_name.scripts import ...)
# # For simplicity here, we assume scripts can find each other if run_pipeline.py is in Scripts.

# def run_script(script_name, stage_name):
#     """Helper function to run a python script and check for errors."""
#     try:
#         print(f"\n----- Running Stage: {stage_name} ({script_name}) -----")
#         # Ensure we're calling python interpreter correctly, sys.executable is a good choice
#         result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
#         print(result.stdout)
#         if result.stderr:
#             print(f"--- Stderr from {script_name} ---")
#             print(result.stderr)
#         print(f"----- Stage: {stage_name} Completed -----")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"----- Error in Stage: {stage_name} ({script_name}) -----")
#         print(f"Return code: {e.returncode}")
#         print("--- Stdout ---")
#         print(e.stdout)
#         print("--- Stderr ---")
#         print(e.stderr)
#         print(f"----- Stage: {stage_name} Failed -----")
#         return False
#     except FileNotFoundError:
#         print(f"----- Error: Script {script_name} not found. -----")
#         return False


# def main_pipeline(mode="train_and_predict"):
#     """
#     Runs the entire data pipeline.
#     mode: "train_and_predict" (full run), "predict_only" (uses existing model)
#     """
#     print("Starting Flight Delay Prediction Pipeline...")

#     # Define paths (assuming this script is in the 'Scripts' directory)
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     database_dir = os.path.join(project_root, "Database")
#     dataset_dir = os.path.join(project_root, "Dataset")
#     models_dir = os.path.join(project_root, "Models")
#     scripts_dir = os.path.join(project_root, "Scripts") # or os.path.dirname(os.path.abspath(__file__))

#     db_path = os.path.join(database_dir, "flight_data.db")
    
#     # --- Ensure critical directories exist ---
#     os.makedirs(database_dir, exist_ok=True)
#     os.makedirs(models_dir, exist_ok=True)

#     # --- Define script paths ---
#     create_db_script = os.path.join(scripts_dir, "create_database.py")
#     # load_data.py, feature_engineering.py, preprocess.py are mostly imported as modules now
#     # But if they had standalone operations needed by the pipeline, they could be called.
#     train_model_script = os.path.join(scripts_dir, "train_model.py")
#     make_predictions_script = os.path.join(scripts_dir, "make_predictions.py")


#     if mode == "train_and_predict":
#         # 1. Create and Populate Database
#         # This script is standalone as it takes CSV paths from within itself.
#         if not run_script(create_db_script, "Database Creation"): return

#         # The subsequent steps (load, feature eng, preprocess, train) are more complex
#         # than simple script calls if they pass dataframes between them.
#         # The provided structure of individual .py files often means they are either:
#         #   a) Standalone callable scripts that read/write intermediate files (not ideal for dataframes)
#         #   b) Modules with functions that are imported and called by a master script (like this one).
#         #
#         # Your current structure (e.g., preprocess.py saving preprocessor.pkl, train_model.py saving model)
#         # leans towards being callable components.
#         # The `if __name__ == '__main__':` blocks in your scripts are for testing them individually.
#         # For the pipeline, we'll import and call their main functions.

#         print("\n----- Running Stage: Data Loading, Feature Engineering, Preprocessing, and Model Training -----")
#         # This integrated step is because preprocess.py and train_model.py need the dataframes.
#         # It's hard to pass dataframes via subprocess calls without saving to disk between each.
#         # So, we'll call their main functions programmatically here.

#         try:
#             from load_data import load_data_from_db
#             from feature_engineering import engineer_features
#             from preprocess import preprocess_data
#             from train_model import train_model # Assuming train_model takes X, y and model_dir

#             print("Step 1: Loading data...")
#             flights_raw_df, airports_raw_df, airlines_raw_df = load_data_from_db(db_path)
#             if flights_raw_df.empty:
#                 print("Failed to load data. Aborting training pipeline.")
#                 return

#             print("Step 2: Engineering features...")
#             engineered_df = engineer_features(flights_raw_df, airports_raw_df, airlines_raw_df)
            
#             # Ensure 'AIRLINE_x' or 'AIRLINE' is correctly handled for preprocess.py
#             # This check can be done here or made more robust within preprocess.py
#             if 'AIRLINE_x' not in engineered_df.columns and 'AIRLINE' in engineered_df.columns:
#                  print("Pipeline Info: Using 'AIRLINE' for preprocessing as 'AIRLINE_x' is not present.")
#             elif 'AIRLINE_x' in engineered_df.columns:
#                  print("Pipeline Info: Using 'AIRLINE_x' for preprocessing.")
#             else:
#                 print("Pipeline Warning: Neither 'AIRLINE_x' nor 'AIRLINE' found for passthrough encoding in engineered_df.")


#             print("Step 3: Preprocessing data and fitting preprocessor...")
#             preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")
#             X_processed, y_target = preprocess_data(engineered_df, preprocessor_path=preprocessor_save_path, fit_preprocessor=True)

#             print("Step 4: Training model...")
#             train_model(X_processed, y_target, models_dir) # train_model saves the model and scalers

#             print("----- Stage: Data Prep & Model Training Completed -----")

#         except ImportError as e:
#             print(f"Error importing a necessary module: {e}. Ensure scripts are in Python path.")
#             return
#         except Exception as e:
#             print(f"Error during integrated data processing and training stage: {e}")
#             import traceback
#             traceback.print_exc()
#             return

#     elif mode == "predict_only":
#         print("Skipping training stages. Proceeding to prediction with existing model.")
#         # Ensure model and preprocessors exist
#         if not os.path.exists(os.path.join(models_dir, "flight_delay_model.keras")) or \
#            not os.path.exists(os.path.join(models_dir, "preprocessor.pkl")):
#             print("Error: Model or preprocessor not found. Cannot run in 'predict_only' mode without prior training.")
#             return
#     else:
#         print(f"Unknown mode: {mode}")
#         return

#     # 5. Make Predictions (common to both modes, assuming model exists if predict_only)
#     # This also requires data loading and feature engineering for the prediction set.
#     # For simplicity, let's assume make_predictions.py handles loading its own "new" data for now,
#     # or we pass a specific dataset to it.
#     # The `make_predictions.py` example uses a sample from the DB.
#     # In a real scenario, you'd point it to a new data source or a specific test set.

#     print("\n----- Running Stage: Making Predictions -----")
#     try:
#         from load_data import load_data_from_db
#         from feature_engineering import engineer_features
#         from make_predictions import make_predictions as run_prediction_logic
#         import sqlite3
#         import pandas as pd

#         # Load data for prediction (e.g., the last N flights, or a specific test set)
#         # For this example, let's use a small sample from the DB again, similar to make_predictions' __main__
#         # This part should be adapted to your actual prediction input source.
#         print("Loading data for prediction...")
#         conn_pred_data = sqlite3.connect(db_path)
#         try:
#             # Example: predict for a few recent flights not yet in predictions table
#             # This query is just an example placeholder.
#             prediction_input_query = """
#             SELECT f.* FROM flights f
#             LEFT JOIN flight_predictions fp ON f.flight_id = fp.flight_id
#             WHERE fp.flight_id IS NULL 
#             ORDER BY f.A_YEAR DESC, f.A_MONTH DESC, f.A_DAY DESC, f.SCHEDULED_DEPARTURE DESC
#             LIMIT 10;
#             """
#             # Or, for a consistent test, use a fixed set of flight_ids or a specific CSV.
#             # For this run_pipeline, let's use a query that just gets a few random ones for demonstration.
#             prediction_input_query_demo = "SELECT * FROM flights ORDER BY RANDOM() LIMIT 5"

#             new_flights_to_predict_df = pd.read_sql_query(prediction_input_query_demo, conn_pred_data)
#             if new_flights_to_predict_df.empty:
#                 print("No new data found to make predictions on based on the example query.")
#             else:
#                 print(f"Loaded {len(new_flights_to_predict_df)} flights for prediction.")
#                 airports_pred_df = pd.read_sql_query("SELECT * FROM airports", conn_pred_data)
#                 airlines_pred_df = pd.read_sql_query("SELECT * FROM airlines", conn_pred_data)
                
#                 print("Engineering features for prediction data...")
#                 engineered_pred_input_df = engineer_features(new_flights_to_predict_df, airports_pred_df, airlines_pred_df)
                
#                 print("Running prediction logic...")
#                 preprocessor_load_path = os.path.join(models_dir, "preprocessor.pkl")
#                 run_prediction_logic(engineered_pred_input_df, models_dir, preprocessor_load_path, db_path)
#         finally:
#             conn_pred_data.close()
        
#         print("----- Stage: Making Predictions Completed -----")

#     except ImportError as e:
#         print(f"Error importing a necessary module for prediction: {e}.")
#     except Exception as e:
#         print(f"Error during prediction stage: {e}")
#         import traceback
#         traceback.print_exc()
#         return

#     print("\nFlight Delay Prediction Pipeline Finished Successfully.")


# if __name__ == "__main__":
#     # To run the full pipeline (create DB, train, predict):
#     # main_pipeline(mode="train_and_predict")

#     # To run only prediction (assuming model and preprocessors exist):
#     main_pipeline(mode="predict_only")

# Scripts/run_pipeline.py
import os
import subprocess
import sys
import argparse # For command-line arguments
import pandas as pd
import sqlite3

# run_script function (can remain as is)
def run_script(script_name, stage_name):
    """Helper function to run a python script and check for errors."""
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
        # ... (error printing) ...
        return False
    except FileNotFoundError:
        print(f"----- Error: Script {script_name} not found. -----")
        return False

def main():
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Pipeline")
    parser.add_argument("action", choices=["train", "prediction"], 
                        help="Specify 'train' to train the model and save test data, or 'prediction' to predict on saved test data.")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    database_dir = os.path.join(project_root, "Database")
    models_dir = os.path.join(project_root, "Models")
    scripts_dir = os.path.join(project_root, "Scripts") # Should be current dir if running from Scripts
    
    db_path = os.path.join(database_dir, "flight_data.db")
    
    os.makedirs(database_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if args.action == "train":
        print("Starting 'train' workflow...")
        create_db_script = os.path.join(scripts_dir, "create_database.py")
        if not run_script(create_db_script, "Database Creation"): return

        print("\n----- Running Stage: Data Loading, Feature Engineering, Preprocessing, Model Training, and Saving Test Data -----")
        try:
            from load_data import load_data_from_db
            from feature_engineering import engineer_features
            from preprocess import preprocess_data # preprocess_data returns X_final, y_final, flight_ids_final
            from train_model import train_model # train_model will now save X_test, y_test to DB

            print("Step 1: Loading data...")
            flights_raw_df, airports_raw_df, airlines_raw_df = load_data_from_db(db_path)
            if flights_raw_df.empty:
                print("Failed to load data. Aborting training pipeline.")
                return

            print("Step 2: Engineering features...")
            # Ensure flight_id is in engineered_df for preprocess_data
            engineered_df = engineer_features(flights_raw_df, airports_raw_df, airlines_raw_df)
            if 'flight_id' not in engineered_df.columns:
                # This should not happen if flights_raw_df has flight_id from create_database.py
                print("CRITICAL ERROR: 'flight_id' missing after feature engineering.")
                return


            print("Step 3: Preprocessing data (fitting preprocessor)...")
            preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")
            # preprocess_data takes the full engineered_df (including target and flight_id)
            # and returns processed X, original scale y, and flight_ids for the WHOLE dataset
            X_processed_full, y_target_full, flight_ids_full = preprocess_data(
                engineered_df.copy(), # Pass a copy to avoid side effects
                preprocessor_path=preprocessor_save_path, 
                fit_preprocessor=True
            )

            print("Step 4: Training model and saving test set to DB...")
            # train_model will internally split X_processed_full, y_target_full, flight_ids_full
            # into its own train/val/test, then save its "test" portion to DB.
            train_model(X_processed_full, y_target_full, flight_ids_full, models_dir, db_path) 

            print("----- 'train' workflow completed. -----")

        except ImportError as e:
            print(f"ImportError in 'train' workflow: {e}")
        except Exception as e:
            print(f"Exception in 'train' workflow: {e}")
            import traceback
            traceback.print_exc()

    elif args.action == "prediction":
        print("Starting 'prediction' workflow...")
        try:
            from predict_on_saved_test import predict_and_evaluate # We'll create this new script/function
            
            print("Running prediction and evaluation on saved test data...")
            predict_and_evaluate(models_dir, db_path)
            
            print("----- 'prediction' workflow completed. -----")

        except ImportError as e:
            print(f"ImportError in 'prediction' workflow: {e}")
        except Exception as e:
            print(f"Exception in 'prediction' workflow: {e}")
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    main()