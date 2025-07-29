# Doesn't use in pipeline and just for completely new incoming data
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import sqlite3

def make_predictions(input_data_df, model_dir, preprocessor_path, output_db_path):
    print("Starting prediction process...")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    ct_preprocessor = joblib.load(preprocessor_path)
    print("Loaded ColumnTransformer preprocessor.")

    scaler_x_path = os.path.join(model_dir, "scaler_x.pkl")
    if not os.path.exists(scaler_x_path):
        raise FileNotFoundError(f"Numeric feature scaler (scaler_x.pkl) not found in {model_dir}")
    scaler_x = joblib.load(scaler_x_path)
    print("Loaded numeric feature scaler (scaler_x).")

    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Target scaler (scaler_y.pkl) not found in {model_dir}")
    scaler_y = joblib.load(scaler_y_path)
    print("Loaded target scaler (scaler_y).")
    
    label_encoders_path = os.path.join(model_dir, "label_encoders.pkl")
    label_encoders = {}
    label_encoders = joblib.load(label_encoders_path)

    X_transformed_np = ct_preprocessor.transform(input_data_df)

    feature_names_out = ct_preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed_np, columns=feature_names_out, index=input_data_df.index)

    for col_original_name, le in label_encoders.items():
        if col_original_name in X_transformed_df.columns:
            try:
                X_transformed_df[col_original_name] = X_transformed_df[col_original_name].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1) # -1 for unknown
            except Exception as e:
                print(f"Warning: Could not label encode {col_original_name}. {e}. Skipping or using default.")
                X_transformed_df[col_original_name] = -1 
        else:
             print(f"Warning: Column {col_original_name} for label encoding not found in transformed data.")

    num_transform_cols_pred = [col for col in X_transformed_df.columns if col.startswith('num__') and col != 'num__ARRIVAL_DELAY'] # Exclude target here
    for col in num_transform_cols_pred:
        X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').astype("float32")

    cat_transform_cols_pred = [col for col in X_transformed_df.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]
    for col in cat_transform_cols_pred:
        X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').fillna(-1).astype("int32")

    cols_to_drop_for_pred = []
    if 'num__ARRIVAL_DELAY' in X_transformed_df.columns:
        cols_to_drop_for_pred.append('num__ARRIVAL_DELAY')
    
    flight_id_col_processed_pred = [col for col in X_transformed_df.columns if 'flight_id' in col and 'passthrough_unchanged__' in col]
    if flight_id_col_processed_pred:
        cols_to_drop_for_pred.extend(flight_id_col_processed_pred)
    
    if cols_to_drop_for_pred:
        X_final_pred = X_transformed_df.drop(columns=cols_to_drop_for_pred, errors='ignore')
    else:
        X_final_pred = X_transformed_df.copy()

    numeric_cols_model_input = [col for col in X_final_pred.columns if col.startswith('num__')]
    cat_cols_model_input = [col for col in X_final_pred.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]


    if not numeric_cols_model_input:
        print("Warning: No numeric columns found for scaling in prediction input.")
        X_pred_num_scaled = np.array([]).reshape(len(X_final_pred), 0) # Empty array with correct number of rows
    else:
        X_pred_num_scaled = scaler_x.transform(X_final_pred[numeric_cols_model_input])

    cat_inputs_pred = {}
    if cat_cols_model_input:
        for col in cat_cols_model_input:
            if col in X_final_pred:
                cat_inputs_pred[f"{col}_inp"] = X_final_pred[col].fillna(-1).astype("int32").values
            else:
                raise ValueError(f"Categorical column {col} expected for model input not found in preprocessed prediction data.")
    else:
        print("No categorical features for model input.")


    # Load Model and Predict 
    model_path = os.path.join(model_dir, "flight_delay_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    model = load_model(model_path, compile=True) 
    print("Loaded trained Keras model.")

    prediction_model_inputs = {"num_inp": X_pred_num_scaled}
    if cat_inputs_pred: prediction_model_inputs.update(cat_inputs_pred)

    predicted_delay_scaled = model.predict(prediction_model_inputs).flatten()

    predicted_delay_actual = scaler_y.inverse_transform(predicted_delay_scaled.reshape(-1, 1)).flatten()
    
    predictions_df = pd.DataFrame({
        'flight_id': input_data_df['flight_id'],
        'predicted_arrival_delay': predicted_delay_actual
    })
    print("Predictions generated.")

    # Save Predictions to Database
    try:
        con = sqlite3.connect(output_db_path)
        cur = con.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS flight_predictions (
            flight_id INTEGER PRIMARY KEY,
            predicted_arrival_delay REAL,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id)
        );
        """)
        predictions_df.to_sql("flight_predictions", con, if_exists="append", index=False) 
        con.commit()
        print(f"Predictions saved to 'flight_predictions' table in {output_db_path}")
    except Exception as e:
        print(f"Error saving predictions to database: {e}")
        if con: con.rollback()
    finally:
        if con: con.close()

    return predictions_df