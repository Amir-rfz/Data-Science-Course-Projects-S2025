# Scripts/make_predictions.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import sqlite3

def make_predictions(input_data_df, model_dir, preprocessor_path, output_db_path):
    """
    Makes predictions on new data using the trained model and preprocessors.
    Saves predictions back to the database.
    'input_data_df' should be a DataFrame with the same raw structure as the training input
    before feature engineering (e.g., loaded from new CSV or database table for prediction).
    """
    print("Starting prediction process...")

    # --- 1. Load Preprocessing Tools ---
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
    if os.path.exists(label_encoders_path):
        label_encoders = joblib.load(label_encoders_path)
        print("Loaded label encoders.")
    else:
        print("Warning: Label encoders file not found. Categorical encoding might fail or be inconsistent if new categories appear.")


    # --- 2. Preprocess Input Data ---
    # The input_data_df needs to go through the SAME feature engineering and preprocessing steps
    # as the training data. This implies that `feature_engineering.py` and parts of
    # `preprocess.py` need to be callable for new data.

    # For this example, let's assume `input_data_df` is already feature engineered
    # and only needs the ColumnTransformer and LabelEncoding part from `preprocess_data`.
    # In a full pipeline, you'd call:
    #   engineered_input_df = feature_engineering.engineer_features(input_data_df, airports_df, airlines_df) # (need airports/airlines too)
    # Then pass engineered_input_df to a simplified preprocessing function.

    # Simplified preprocessing for prediction:
    # The ColumnTransformer (ct_preprocessor) expects specific columns.
    # We assume input_data_df has these columns from prior feature engineering.
    
    # Impute and transform using the loaded ColumnTransformer
    # The CT will select the columns it was trained on.
    try:
        X_transformed_np = ct_preprocessor.transform(input_data_df)
    except ValueError as e:
        print(f"Error transforming input_data_df with ColumnTransformer: {e}")
        print(f"Make sure input_data_df has all columns the preprocessor was fit on, with correct dtypes.")
        print(f"Input columns: {input_data_df.columns.tolist()}")
        # You might need to access ct_preprocessor.feature_names_in_ or similar to debug
        return

    feature_names_out = ct_preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed_np, columns=feature_names_out, index=input_data_df.index)

    # Apply loaded LabelEncoders to the relevant passthrough columns
    for col_original_name, le in label_encoders.items():
        # col_original_name here is the name used when saving, e.g., 'passthrough_label__AIRLINE_x'
        if col_original_name in X_transformed_df.columns:
            try:
                # Handle unseen labels by assigning a specific code (e.g., -1 or len(classes))
                # This requires modifying the LabelEncoder or handling it here.
                # A simple approach: if a value is not in le.classes_, map it to a default.
                X_transformed_df[col_original_name] = X_transformed_df[col_original_name].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1) # -1 for unknown
            except Exception as e:
                print(f"Warning: Could not label encode {col_original_name}. {e}. Skipping or using default.")
                X_transformed_df[col_original_name] = -1 # Or some other placeholder
        else:
             print(f"Warning: Column {col_original_name} for label encoding not found in transformed data.")


    # Data type conversions (similar to preprocess.py)
    num_transform_cols_pred = [col for col in X_transformed_df.columns if col.startswith('num__') and col != 'num__ARRIVAL_DELAY'] # Exclude target here
    for col in num_transform_cols_pred:
        X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').astype("float32")

    cat_transform_cols_pred = [col for col in X_transformed_df.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]
    for col in cat_transform_cols_pred:
        X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').fillna(-1).astype("int32")


    # Drop 'num__ARRIVAL_DELAY' if it's present (it shouldn't be if this is purely for prediction on new data)
    # Also drop flight_id if it was passed through by CT
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
        
    # Ensure column order and presence matches training (handled by ColumnTransformer if used correctly)
    # Identify numeric and categorical columns for the model input preparation
    numeric_cols_model_input = [col for col in X_final_pred.columns if col.startswith('num__')]
    cat_cols_model_input = [col for col in X_final_pred.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]


    # --- 3. Scale Numeric Features ---
    if not numeric_cols_model_input:
        print("Warning: No numeric columns found for scaling in prediction input.")
        X_pred_num_scaled = np.array([]).reshape(len(X_final_pred), 0) # Empty array with correct number of rows
    else:
        # Ensure all numeric columns expected by scaler_x are present
        # scaler_x was fit on a specific set of columns (e.g., ['num__DISTANCE_KM', ...])
        # We need to ensure X_final_pred[numeric_cols_model_input] has these exact columns in the right order.
        # This is usually guaranteed if ct_preprocessor output names are consistent and numeric_cols_model_input is derived from them.
        X_pred_num_scaled = scaler_x.transform(X_final_pred[numeric_cols_model_input])


    # --- 4. Prepare Categorical Inputs ---
    cat_inputs_pred = {}
    if cat_cols_model_input:
        for col in cat_cols_model_input:
            if col in X_final_pred:
                cat_inputs_pred[f"{col}_inp"] = X_final_pred[col].fillna(-1).astype("int32").values
            else:
                raise ValueError(f"Categorical column {col} expected for model input not found in preprocessed prediction data.")
    else:
        print("No categorical features for model input.")


    # --- 5. Load Model and Predict ---
    model_path = os.path.join(model_dir, "flight_delay_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    model = load_model(model_path, compile=True) # compile=True is safer if custom objects were used, but Adam/mse/mae are standard
    print("Loaded trained Keras model.")

    prediction_model_inputs = {"num_inp": X_pred_num_scaled}
    if cat_inputs_pred: prediction_model_inputs.update(cat_inputs_pred)

    if not X_pred_num_scaled.any() and not cat_inputs_pred: # Check if both are effectively empty
         print("No data to predict. Aborting.")
         return pd.DataFrame() # Return empty DataFrame


    predicted_delay_scaled = model.predict(prediction_model_inputs).flatten()

    # --- 6. Inverse Transform Predictions ---
    predicted_delay_actual = scaler_y.inverse_transform(predicted_delay_scaled.reshape(-1, 1)).flatten()
    
    predictions_df = pd.DataFrame({
        'flight_id': input_data_df['flight_id'], # Assuming 'flight_id' is in the original input_data_df
        'predicted_arrival_delay': predicted_delay_actual
    })
    print("Predictions generated.")

    # --- 7. Save Predictions to Database ---
    # The project description implies saving predictions to a *new* or *existing relevant* table.
    # Let's create a new table 'flight_predictions'.
    try:
        con = sqlite3.connect(output_db_path)
        cur = con.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS flight_predictions (
            flight_id INTEGER PRIMARY KEY,
            predicted_arrival_delay REAL,
            prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id)
        );
        """)
        # Use INSERT OR REPLACE to update if flight_id already exists, or insert new
        predictions_df.to_sql("flight_predictions", con, if_exists="append", index=False) # 'append' assumes flight_id is unique, or use 'replace' with caution
        con.commit()
        print(f"Predictions saved to 'flight_predictions' table in {output_db_path}")
    except Exception as e:
        print(f"Error saving predictions to database: {e}")
        if con: con.rollback()
    finally:
        if con: con.close()

    return predictions_df


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "Database/flight_data.db")
    models_dir = os.path.join(project_root, "Models")
    preprocessor_load_path = os.path.join(models_dir, "preprocessor.pkl")

    # For testing make_predictions, we need some sample input data.
    # This data should be in the raw format *before* feature engineering,
    # as if it's new unseen data.
    # Let's try to load a few samples from the original flights table for this test.
    
    conn_test = sqlite3.connect(db_path)
    try:
        # Load a small sample of raw flight data to simulate new data
        # We need to also load airports and airlines for the feature engineering step
        sample_flights_df = pd.read_sql_query("SELECT * FROM flights ORDER BY RANDOM() LIMIT 5", conn_test)
        airports_df_test = pd.read_sql_query("SELECT * FROM airports", conn_test)
        airlines_df_test = pd.read_sql_query("SELECT * FROM airlines", conn_test)
    finally:
        conn_test.close()

    if not sample_flights_df.empty:
        print("Sample raw data loaded for prediction test:")
        print(sample_flights_df[['flight_id', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE']])

        # 1. Perform Feature Engineering on this sample data
        from feature_engineering import engineer_features # Assuming this is in the same dir or python path
        
        # Ensure engineer_features can handle the columns from the raw DB load
        # It expects A_YEAR, A_MONTH, A_DAY, etc.
        # The DB load should provide these.
        try:
            engineered_sample_df = engineer_features(sample_flights_df, airports_df_test, airlines_df_test)
            print("\nEngineered sample data for prediction:")
            print(engineered_sample_df[['flight_id', 'DISTANCE_KM', 'AIRLINE_7D_MEAN', 'DEP_HOUR', 'DAY_OF_WEEK']].head())

            # 2. Now, call make_predictions with this engineered data
            # Note: The `make_predictions` function currently assumes that the `input_data_df` argument
            # it receives is *already feature engineered*. It then applies the ColumnTransformer.
            # This matches the flow: raw -> feature_eng -> CT_transform -> scale -> predict
            
            # Ensure the engineered_sample_df has 'flight_id' for merging predictions later
            if 'flight_id' not in engineered_sample_df.columns:
                 # This should not happen if engineer_features preserves or creates it from the input `sample_flights_df`
                print("Error: 'flight_id' missing from engineered sample data.")
            else:
                predictions_output = make_predictions(engineered_sample_df, models_dir, preprocessor_load_path, db_path)
                if not predictions_output.empty:
                    print("\nPredictions output:")
                    print(predictions_output)

                    # Verify in DB
                    conn_verify = sqlite3.connect(db_path)
                    try:
                        preds_from_db = pd.read_sql_query(f"SELECT * FROM flight_predictions WHERE flight_id IN ({','.join(map(str, predictions_output.flight_id.tolist()))})", conn_verify)
                        print("\nPredictions retrieved from database:")
                        print(preds_from_db)
                    finally:
                        conn_verify.close()

        except Exception as e:
            print(f"An error occurred during the prediction test: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("No sample data loaded from DB, skipping prediction test.")