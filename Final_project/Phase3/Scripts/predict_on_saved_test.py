# Scripts/predict_on_saved_test.py
import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Import smape if you defined it in a utils.py or here
# from utils import smape # Assuming smape is in utils
def smape(y_true, y_pred): # Or define it here if not in utils
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(num / np.maximum(den, 1e-8)) * 100


def predict_and_evaluate(model_dir, db_path):
    print("Starting prediction and evaluation on saved test data...")

    # --- 1. Load Saved Test Data from Database ---
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        # Load y_test
        df_y_test = pd.read_sql_query("SELECT flight_id, actual_arrival_delay FROM saved_y_test", conn)
        if df_y_test.empty:
            print("No data found in 'saved_y_test' table. Aborting prediction.")
            return
        
        # Load X_test_processed
        # Note: Reading the whole table. For very large test sets, consider chunking or specific flight_ids.
        df_X_test_processed = pd.read_sql_query("SELECT * FROM saved_processed_X_test", conn)
        if df_X_test_processed.empty:
            print("No data found in 'saved_processed_X_test' table. Aborting prediction.")
            return

        # Merge to ensure alignment (though they should be if saved correctly by flight_id)
        # Set flight_id as index for easy alignment before processing
        df_y_test = df_y_test.set_index('flight_id')
        ids_test_loaded = df_X_test_processed['flight_id'] # Keep IDs for saving predictions
        df_X_test_processed = df_X_test_processed.set_index('flight_id')
        
        # Align X and y based on common flight_ids (index)
        common_flight_ids = df_X_test_processed.index.intersection(df_y_test.index)
        df_X_test_processed = df_X_test_processed.loc[common_flight_ids]
        df_y_test_aligned = df_y_test.loc[common_flight_ids]['actual_arrival_delay']
        ids_test_loaded = ids_test_loaded[ids_test_loaded.isin(common_flight_ids)]


        if df_X_test_processed.empty:
            print("No common flight_ids between saved X_test and y_test. Aborting.")
            return
            
        print(f"Loaded {len(df_X_test_processed)} records from saved test set.")

    except Exception as e:
        print(f"Error loading test data from database: {e}")
        if conn: conn.close()
        return
    # finally: # Keep conn open for saving predictions later if structure allows, or reopen
    #     if conn: conn.close()


    # --- 2. Load Model and Preprocessors ---
    try:
        model_path = os.path.join(model_dir, "flight_delay_model.keras")
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")

        scaler_x_path = os.path.join(model_dir, "scaler_x.pkl")
        scaler_x = joblib.load(scaler_x_path)
        print(f"Loaded scaler_x from {scaler_x_path}")

        scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
        scaler_y = joblib.load(scaler_y_path)
        print(f"Loaded scaler_y from {scaler_y_path}")
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        return

    # --- 3. Prepare Loaded X_test_processed for Prediction ---
    # df_X_test_processed already has the ColumnTransformer's output structure
    numeric_cols_pred = [col for col in df_X_test_processed.columns if col.startswith('num__')]
    cat_cols_pred = [col for col in df_X_test_processed.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]

    # Apply scaler_x to numeric features
    X_test_num_scaled = scaler_x.transform(df_X_test_processed[numeric_cols_pred])

    # Prepare categorical inputs (they are already encoded integers)
    cat_inputs_test_pred = {}
    for col in cat_cols_pred:
        cat_inputs_test_pred[f"{col}_inp"] = df_X_test_processed[col].fillna(-1).astype("int32").values
        
    model_inputs_pred = {"num_inp": X_test_num_scaled, **cat_inputs_test_pred}

    # --- 4. Make Predictions ---
    y_pred_scaled = model.predict(model_inputs_pred).flatten()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    print("Predictions generated.")

    # --- 5. Evaluate Metrics ---
    y_true_original = df_y_test_aligned.values # This is the actual_arrival_delay series aligned with X_test

    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)
    smape_val = smape(y_true_original, y_pred_original)

    print("\n----- Prediction Metrics on Saved Test Data (Original Scale) -----")
    print(f"MSE   : {mse:,.3f}")
    print(f"RMSE  : {rmse:,.3f}")
    print(f"MAE   : {mae:,.3f}")
    print(f"R²    : {r2:,.4f}")
    print(f"SMAPE : {smape_val:.2f}%")

    # --- 6. Save Predictions to Database ---
    df_predictions = pd.DataFrame({
        'flight_id': ids_test_loaded.values, # Use the flight_ids corresponding to the X_test data
        'predicted_arrival_delay': y_pred_original
    })

    # Use a new connection object specifically for this database operation
    conn_save = None 
    try:
        conn_save = sqlite3.connect(db_path) # Establish a new connection for saving
        cur = conn_save.cursor()
        
        # Create pipeline_test_predictions table
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_test_predictions (
            flight_id INTEGER PRIMARY KEY,
            predicted_arrival_delay REAL,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id)
        );
        """)
        # Clear previous predictions for this table (optional, based on your requirements)
        cur.execute("DELETE FROM pipeline_test_predictions;") 
        df_predictions.to_sql("pipeline_test_predictions", conn_save, if_exists="append", index=False)
        conn_save.commit()
        print(f"Saved predictions ( {len(df_predictions)} records) to 'pipeline_test_predictions' table.")
    except Exception as e:
        print(f"Error saving predictions to database: {e}")
        if conn_save: 
            conn_save.rollback()
    finally:
        if conn_save: 
            conn_save.close() # Ensure this specific connection is closed

if __name__ == '__main__':
    # For standalone testing of predict_on_saved_test.py
    print("predict_on_saved_test.py standalone execution (for testing purposes):")
    project_root_main = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir_main = os.path.join(project_root_main, "Models")
    db_path_main = os.path.join(project_root_main, "Database/flight_data.db")
    
    # Ensure test data exists in DB and model/scalers are in Models dir
    if os.path.exists(models_dir_main) and os.path.exists(db_path_main):
        # Check if essential files/tables exist before calling
        # This is just a basic check; the function itself handles more specific file/table not found errors
        predict_and_evaluate(models_dir_main, db_path_main)
    else:
        print("Models directory or database path not found. Cannot run standalone test.")