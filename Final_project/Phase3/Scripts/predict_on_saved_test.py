import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from tensorflow.keras.models import load_model 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow 
import tempfile 

def smape(y_true, y_pred):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(num / np.maximum(den, 1e-8)) * 100

def predict_and_evaluate(model_dir, db_path):
    print("Starting prediction and evaluation...")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df_y_test = pd.read_sql_query("SELECT flight_id, actual_arrival_delay FROM saved_y_test", conn)
        df_X_test_processed_db = pd.read_sql_query("SELECT * FROM saved_processed_X_test", conn) 
        
        df_y_test = df_y_test.set_index('flight_id')
        ids_test_loaded = df_X_test_processed_db['flight_id']
        df_X_test_processed = df_X_test_processed_db.set_index('flight_id') 
        
        common_flight_ids = df_X_test_processed.index.intersection(df_y_test.index)
        df_X_test_processed = df_X_test_processed.loc[common_flight_ids]
        df_y_test_aligned = df_y_test.loc[common_flight_ids]['actual_arrival_delay']
        ids_test_loaded = ids_test_loaded[ids_test_loaded.isin(common_flight_ids)]
        print(f"Loaded {len(df_X_test_processed)} records from saved test set.")
        mlflow.log_param("prediction_data_status", "loaded_successfully")
        mlflow.log_param("prediction_test_set_size", len(df_X_test_processed))
    except Exception as e:
        print(f"Error loading test data: {e}"); mlflow.log_param("prediction_data_status", f"error_{e}"); return
    finally:
        if conn: conn.close()


    try:
        local_model_path = os.path.join(model_dir, "flight_delay_model_local.keras") 
        if not os.path.exists(local_model_path):
            print(f"Local model copy not found at {local_model_path}. Attempting to load from MLflow if URI known.")
            raise FileNotFoundError(f"Local model not found at {local_model_path}. Update logic to load from MLflow registry or ensure local save.")
        model = load_model(local_model_path)
        print(f"Loaded model from {local_model_path}")
        mlflow.set_tag("model_source", local_model_path) 

        scaler_x_path = os.path.join(model_dir, "scaler_x.pkl")
        scaler_x = joblib.load(scaler_x_path)
        mlflow.set_tag("scaler_x_source", scaler_x_path)

        scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
        scaler_y = joblib.load(scaler_y_path)
        mlflow.set_tag("scaler_y_source", scaler_y_path)

    except Exception as e:
        print(f"Error loading model or scalers: {e}"); mlflow.log_param("prediction_artifact_load_status", f"error_{e}"); return

    numeric_cols_pred = [col for col in df_X_test_processed.columns if col.startswith('num__')]
    cat_cols_pred = [col for col in df_X_test_processed.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]
    X_test_num_scaled = scaler_x.transform(df_X_test_processed[numeric_cols_pred])
    cat_inputs_test_pred = {}
    for col in cat_cols_pred: cat_inputs_test_pred[f"{col}_inp"] = df_X_test_processed[col].fillna(-1).astype("int32").values
    model_inputs_pred = {"num_inp": X_test_num_scaled, **cat_inputs_test_pred}
    y_pred_scaled = model.predict(model_inputs_pred).flatten()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    y_true_original = df_y_test_aligned.values
    metrics_to_log = {
        "test_mse": mean_squared_error(y_true_original, y_pred_original),
        "test_rmse": np.sqrt(mean_squared_error(y_true_original, y_pred_original)),
        "test_mae": mean_absolute_error(y_true_original, y_pred_original),
        "test_r2": r2_score(y_true_original, y_pred_original),
        "test_smape": smape(y_true_original, y_pred_original)
    }
    mlflow.log_metrics(metrics_to_log)
    print("\n----- Prediction Metrics on Saved Test Data (Logged to MLflow) -----")
    for k, v in metrics_to_log.items(): 
        print(f"{k.replace('_', ' ').title()} : {v:.4f}")


    # Save Predictions to Database and Log as Artifact
    df_predictions = pd.DataFrame({'flight_id': ids_test_loaded.values, 'predicted_arrival_delay': y_pred_original})
    conn_save = None
    try:
        conn_save = sqlite3.connect(db_path) 
        cur = conn_save.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_test_predictions (
            flight_id INTEGER PRIMARY KEY,
            predicted_arrival_delay REAL,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id)
        );
        """)
        cur.execute("DELETE FROM pipeline_test_predictions;"); df_predictions.to_sql("pipeline_test_predictions", conn_save, if_exists="append", index=False)
        conn_save.commit(); print(f"Saved predictions to 'pipeline_test_predictions' table.")
        mlflow.log_param("db_predictions_save_status", "success")

        # Log predictions DataFrame as a CSV 
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp_pred_file:
            df_predictions.to_csv(tmp_pred_file.name, index=False)
            mlflow.log_artifact(tmp_pred_file.name, "predictions_output/test_set_predictions.csv")
        os.remove(tmp_pred_file.name) 
        print("Logged test set predictions as CSV artifact to MLflow.")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        mlflow.log_param("db_predictions_save_status", f"error_{e}")
        if conn_save: conn_save.rollback()
    finally:
        if conn_save: conn_save.close()

