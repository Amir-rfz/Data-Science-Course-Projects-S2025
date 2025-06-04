# # Scripts/preprocess.py
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# import joblib # For saving the preprocessor
# import os

# def preprocess_data(flights_df, preprocessor_path=None, fit_preprocessor=False, test_mode=False):
#     """
#     Preprocesses the flight data: handles missing values, encodes categorical features.
#     Saves the fitted preprocessor if fit_preprocessor is True.
#     Loads and uses a saved preprocessor if fit_preprocessor is False.
#     """
#     print("Starting data preprocessing...")

#     # Imputation based on your original notebook
#     delay_cols = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
#     flights_df[delay_cols] = flights_df[delay_cols].fillna(0)

#     delay_basic_cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY']
#     # For ARRIVAL_DELAY, we only fill if it's not the target in a prediction scenario
#     # In training, ARRIVAL_DELAY is used to create the target, so NaNs there might be significant
#     # However, your original code fills it with 0 before creating features like DELAY_PER_KM
#     # and for the target y, it uses > 15. We'll keep fillna(0) for consistency with your features.
#     flights_df[delay_basic_cols] = flights_df[delay_basic_cols].fillna(0)


#     time_cols = ['DEPARTURE_TIME', 'ARRIVAL_TIME', 'WHEELS_ON', 'WHEELS_OFF',
#                  'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME']

#     for col in time_cols:
#         # Fill NaNs for cancelled flights with -1
#         if 'CANCELLED' in flights_df.columns: # Ensure CANCELLED column exists
#             flights_df.loc[flights_df['CANCELLED'] == 1, col] = flights_df.loc[flights_df['CANCELLED'] == 1, col].fillna(-1)
#         # Fill remaining NaNs (e.g., for non-cancelled flights or if CANCELLED doesn't exist) with median
#         flights_df[col] = flights_df[col].fillna(flights_df[col].median())


#     cat_cols_to_fill_unknown = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'] # Original airline name, not IATA code
#     for col in cat_cols_to_fill_unknown:
#         if col in flights_df.columns:
#              flights_df[col] = flights_df[col].fillna('Unknown')


#     # Drop CANCELLATION_REASON if it exists
#     flights_df = flights_df.drop(columns=['CANCELLATION_REASON'], errors='ignore')

#     # Define feature sets (ensure ARRIVAL_DELAY is in num_feats for the ColumnTransformer)
#     num_feats = ['DISTANCE_KM', 'AIRLINE_7D_MEAN', 'DEP_HOUR', 'DELAY_PER_KM',
#                  'DEPARTURE_DELAY', 'ELAPSED_TIME', 'ARRIVAL_DELAY'] # ARRIVAL_DELAY is needed for transformation and target
#     cat_feats = ['DAY_OF_WEEK', 'WEEK_OF_YEAR', 'IS_WEEKEND'] # These are already numerical after feature eng.
    
#     # These are passthrough and then label encoded
#     passthrough_label_encode_cols = ['AIRLINE_x', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'TAIL_NUMBER']
#     # Suffix '_x' might be added to 'AIRLINE' by the merge in feature_engineering if 'AIRLINE' also existed in airlines_df
#     # Adjust based on actual column names after merge. If it's just 'AIRLINE', use that.
#     # Check if 'AIRLINE_x' exists, otherwise use 'AIRLINE'
#     if 'AIRLINE_x' not in flights_df.columns and 'AIRLINE' in flights_df.columns:
#         passthrough_label_encode_cols = ['AIRLINE' if item == 'AIRLINE_x' else item for item in passthrough_label_encode_cols]


#     unchanged_columns = ['flight_id'] # Columns to keep as is, if any, or handle separately

#     # Ensure all defined columns are present in flights_df
#     all_cols_needed = num_feats + cat_feats + [col for col in passthrough_label_encode_cols if col in flights_df.columns] + unchanged_columns
#     missing_cols = [col for col in all_cols_needed if col not in flights_df.columns and col != 'flight_id'] # flight_id is added later if not present
#     if 'flight_id' not in flights_df.columns and 'flight_id' in unchanged_columns:
#         flights_df.insert(0, "flight_id", range(1, len(flights_df) + 1))


#     if missing_cols:
#         raise ValueError(f"Missing columns required for preprocessing: {missing_cols}. Available columns: {flights_df.columns.tolist()}")


#     # Define pipelines for ColumnTransformer
#     num_pipeline = Pipeline([
#         ('impute', SimpleImputer(strategy='median'))
#         # StandardScaler will be applied separately in train_model.py after splitting data
#     ])

#     cat_pipeline = Pipeline([
#         ('impute', SimpleImputer(strategy='constant', fill_value=-1)), # For DAY_OF_WEEK etc.
#         ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) # though they are already numeric
#     ])


#     # Create the ColumnTransformer
#     # Adjust passthrough_label_encode_cols to only include those present in flights_df
#     final_passthrough_cols = [col for col in passthrough_label_encode_cols if col in flights_df.columns]

#     preprocessor_transformers = [
#         ('num', num_pipeline, num_feats),
#         ('cat', cat_pipeline, cat_feats)
#     ]
#     if final_passthrough_cols: # Only add passthrough if there are columns for it
#         preprocessor_transformers.append(('passthrough_label', 'passthrough', final_passthrough_cols))
#     if unchanged_columns and 'flight_id' in unchanged_columns :
#          preprocessor_transformers.append(('passthrough_unchanged', 'passthrough', ['flight_id']))


#     ct = ColumnTransformer(preprocessor_transformers, remainder='drop') # Drop other columns not specified

#     processed_data_placeholder = np.array([]) # Placeholder for when preprocessor cannot be fit/transformed

#     if fit_preprocessor:
#         print("Fitting preprocessor...")
#         # Ensure all columns exist before fitting
#         cols_for_ct_fit = num_feats + cat_feats + final_passthrough_cols
#         if 'flight_id' in unchanged_columns:
#              cols_for_ct_fit += ['flight_id']
        
#         missing_for_fit = [col for col in cols_for_ct_fit if col not in flights_df.columns]
#         if missing_for_fit:
#             raise ValueError(f"Cannot fit preprocessor. Missing columns: {missing_for_fit}")

#         X_transformed = ct.fit_transform(flights_df[cols_for_ct_fit])
#         os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
#         joblib.dump(ct, preprocessor_path)
#         print(f"Preprocessor fitted and saved to {preprocessor_path}")
#         processed_data_placeholder = X_transformed
#     elif preprocessor_path and os.path.exists(preprocessor_path):
#         print(f"Loading preprocessor from {preprocessor_path}...")
#         ct = joblib.load(preprocessor_path)
#          # Ensure all columns expected by the loaded preprocessor are present
#         # This is a bit tricky as ct.feature_names_in_ is not standard for all versions/transformers
#         # A robust way is to ensure the input df to transform has at least the columns it was fit on.
#         # For simplicity, we assume test_df has the necessary columns.
#         # The ColumnTransformer will select the ones it needs.
#         X_transformed = ct.transform(flights_df) # Pass the whole df, CT will select
#         print("Data transformed using loaded preprocessor.")
#         processed_data_placeholder = X_transformed
#     else:
#         if test_mode: # In test mode, if no preprocessor, we can't proceed
#             raise FileNotFoundError("Preprocessor not found and not in fitting mode for test data.")
#         # If not fitting and no preprocessor path, this implies an issue or a flow where it's not needed yet
#         print("Warning: Preprocessor not fitted or loaded. Returning data after basic imputation only.")
#         # In this case, X_transformed won't be created by ct. We need to construct a compatible DataFrame.
#         # For now, we'll return the partially processed flights_df and handle downstream.
#         # This part might need adjustment based on how `run_pipeline` calls it.
#         # The intention is that `processed_df` should be the output of ColumnTransformer.
#         # If CT didn't run, we need a different approach or ensure it always runs.
#         # For now, if CT doesn't run, the structure after this won't match.
#         # A simple solution for now: if ct wasn't used, just make `processed_df` from `flights_df` with relevant columns
#         # This part requires careful thought about the pipeline flow.
#         # Let's assume for now `fit_preprocessor=True` for training and `preprocessor_path` is valid for testing.
#         # So, X_transformed should always be populated if the function is called correctly in a pipeline.


#     # Reconstruct DataFrame from ColumnTransformer output
#     feature_names = ct.get_feature_names_out()
#     processed_df = pd.DataFrame(processed_data_placeholder, columns=feature_names, index=flights_df.index)


#     # TARGET: Define the target variable (ARRIVAL_DELAY > 15).
#     # This target is for a classification task, but your model predicts regression (ARRIVAL_DELAY itself)
#     # Your original notebook defines y = (flights['ARRIVAL_DELAY'] > 15).astype(int)
#     # but then the NN model predicts the actual ARRIVAL_DELAY (regression).
#     # Let's stick to predicting ARRIVAL_DELAY as per your NN model.
#     # The 'num__ARRIVAL_DELAY' column from the preprocessor IS our target if scaled,
#     # or we can take it from flights_df['ARRIVAL_DELAY'] before scaling if we scale target separately.
#     # Your NN model scales y (y_scaler). So we use 'ARRIVAL_DELAY' from the original data.
#     y_target = flights_df['ARRIVAL_DELAY'].copy() # This is the raw arrival delay

#     # Drop original columns that are now transformed or not needed, except the target and IDs
#     # The ColumnTransformer already handles selecting features, so `processed_df` is X.
#     # We need to handle the label encoding for passthrough columns separately if they weren't dropped
    
#     label_encoders = {}
#     # Apply Label Encoding to the passthrough columns that were designated for it
#     # These columns are now prefixed like 'passthrough_label__AIRLINE_x'
#     for col_original_name in passthrough_label_encode_cols:
#         # Find the column name in processed_df (it will have a prefix)
#         transformed_col_name = None
#         for processed_col_name in processed_df.columns:
#             if col_original_name in processed_col_name and 'passthrough_label__' in processed_col_name:
#                 transformed_col_name = processed_col_name
#                 break
        
#         if transformed_col_name and transformed_col_name in processed_df.columns:
#             # On training, fit and transform. On testing, just transform.
#             if fit_preprocessor:
#                 le = LabelEncoder()
#                 processed_df[transformed_col_name] = le.fit_transform(processed_df[transformed_col_name].astype(str))
#                 label_encoders[transformed_col_name] = le # Save the encoder
#             elif col_original_name in label_encoders: # This assumes label_encoders are passed if not fitting
#                  # This part needs refinement: label_encoders should be loaded if not fitting
#                  # For now, we assume this function is either fitting or the encoders are handled externally for prediction.
#                  # A proper pipeline would save/load these encoders similar to the preprocessor.
#                 print(f"Warning: LabelEncoder for {transformed_col_name} would need to be loaded for test data if not fitting here.")
#                 # Fallback to astype category codes if encoder not available (not robust)
#                 processed_df[transformed_col_name] = processed_df[transformed_col_name].astype('category').cat.codes

#     if fit_preprocessor and label_encoders:
#         # Save label_encoders if any were fit
#         # This path should be managed, e.g., passed as an argument
#         le_path = os.path.join(os.path.dirname(preprocessor_path), "label_encoders.pkl")
#         joblib.dump(label_encoders, le_path)
#         print(f"Label encoders saved to {le_path}")


#     # Drop rows with NaN in TAIL_NUMBER if it's a passthrough column after transformation
#     # Example: 'passthrough_label__TAIL_NUMBER'
#     tail_num_col_in_processed = [col for col in processed_df.columns if 'TAIL_NUMBER' in col and 'passthrough_label__' in col]
#     if tail_num_col_in_processed:
#         processed_df = processed_df.dropna(subset=tail_num_col_in_processed)
#         # The target y_target must be aligned with processed_df
#         y_target = y_target[processed_df.index]


#     # Reset index after potential drops
#     processed_df.reset_index(drop=True, inplace=True)
#     y_target.reset_index(drop=True, inplace=True)


#     # Data type conversions (as per your notebook, ensure correct column names)
#     # Numeric columns from ColumnTransformer (prefixed with 'num__')
#     num_transform_cols = [col for col in processed_df.columns if col.startswith('num__')]
#     for col in num_transform_cols:
#         processed_df[col] = processed_df[col].astype("float32")

#     # Categorical columns from ColumnTransformer (prefixed with 'cat__')
#     # and LabelEncoded columns (prefixed with 'passthrough_label__')
#     cat_transform_cols = [col for col in processed_df.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]
#     for col in cat_transform_cols:
#         processed_df[col] = processed_df[col].astype("int32") # Ordinal and Label Encoded are integers

#     print("Data preprocessing completed.")
    
#     # Return the processed features (X) and the target (y)
#     # The target 'num__ARRIVAL_DELAY' is part of processed_df if 'ARRIVAL_DELAY' was in num_feats
#     # We need to separate it.
#     target_col_name_in_processed = 'num__ARRIVAL_DELAY' # Assuming it's processed by num_pipeline

#     if target_col_name_in_processed not in processed_df.columns:
#         # This case should ideally not happen if ARRIVAL_DELAY is in num_feats.
#         # If it's not, y_target (raw) should be used and scaled separately.
#         # For now, let's assume it is there.
#         raise ValueError(f"Target column {target_col_name_in_processed} not found in processed_df columns: {processed_df.columns}")

#     X_final = processed_df.drop(columns=[target_col_name_in_processed], errors='ignore')
#     y_final = processed_df[target_col_name_in_processed].copy() # This is the (potentially imputed) ARRIVAL_DELAY

#     # If 'passthrough_unchanged__flight_id' exists, drop it from X_final as it's not a feature for the model
#     flight_id_col_processed = [col for col in X_final.columns if 'flight_id' in col and 'passthrough_unchanged__' in col]
#     if flight_id_col_processed:
#         X_final = X_final.drop(columns=flight_id_col_processed)


#     print(f"Final X shape: {X_final.shape}, Final y shape: {y_final.shape}")
#     return X_final, y_final


# if __name__ == '__main__':
#     import os
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     db_path = os.path.join(project_root, "Database/flight_data.db")
#     models_dir = os.path.join(project_root, "Models")
#     preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")

#     from load_data import load_data_from_db
#     from feature_engineering import engineer_features

#     flights_df, airports_df, airlines_df = load_data_from_db(db_path)
#     if not flights_df.empty:
#         engineered_df = engineer_features(flights_df, airports_df, airlines_df)
        
#         # Ensure 'AIRLINE_x' or 'AIRLINE' is correctly handled before preprocessing
#         if 'AIRLINE_x' not in engineered_df.columns and 'AIRLINE' in engineered_df.columns:
#              print("Using 'AIRLINE' for preprocessing as 'AIRLINE_x' is not present.")
#         elif 'AIRLINE_x' in engineered_df.columns:
#              print("Using 'AIRLINE_x' for preprocessing.")
#         else:
#             print("Warning: Neither 'AIRLINE_x' nor 'AIRLINE' found for passthrough encoding.")


#         X_processed, y_target_values = preprocess_data(engineered_df, preprocessor_path=preprocessor_save_path, fit_preprocessor=True)
#         print("\nProcessed X head:")
#         print(X_processed.head())
#         print("\nTarget y head:")
#         print(y_target_values.head())
#         print(f"\nProcessed X dtypes:\n{X_processed.dtypes}")
#         print(f"\nTarget y dtype: {y_target_values.dtype}")

#         # Example of loading and transforming (simulating prediction flow)
#         # This would typically happen in make_predictions.py
#         if os.path.exists(preprocessor_save_path):
#             print("\nSimulating loading preprocessor and transforming new data...")
#             # In a real scenario, new_data would come from load_data -> feature_engineering
#             # For this test, we can reuse engineered_df (or a sample of it)
#             new_data_engineered = engineered_df.sample(n=5, random_state=42).copy()
#             X_new_processed, y_new_target = preprocess_data(new_data_engineered, preprocessor_path=preprocessor_save_path, fit_preprocessor=False, test_mode=True)
#             print("\nProcessed X_new head:")
#             print(X_new_processed.head())
#             print("\nTarget y_new head (would be actuals for evaluation if available, or NaN for prediction):")
#             print(y_new_target.head())


#     else:
#         print("Skipping preprocessing due to empty input dataframes.")

# Scripts/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_data(flights_df, preprocessor_path=None, fit_preprocessor=False, test_mode=False):
    """
    Preprocesses the flight data: handles missing values, encodes categorical features.
    Saves the fitted preprocessor if fit_preprocessor is True.
    Loads and uses a saved preprocessor if fit_preprocessor is False.
    Returns:
        X_final (pd.DataFrame): Processed features.
        y_final (pd.Series): Processed target variable.
        flight_ids_final (pd.Series): Corresponding flight_ids.
    """
    print("Starting data preprocessing...")

    # ... (all your existing imputation and CANCELLATION_REASON drop code remains the same) ...
    # Current imputation logic:
    delay_cols = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
    flights_df[delay_cols] = flights_df[delay_cols].fillna(0)

    delay_basic_cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY']
    flights_df[delay_basic_cols] = flights_df[delay_basic_cols].fillna(0)

    time_cols = ['DEPARTURE_TIME', 'ARRIVAL_TIME', 'WHEELS_ON', 'WHEELS_OFF',
                 'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME']
    for col in time_cols:
        if 'CANCELLED' in flights_df.columns:
            flights_df.loc[flights_df['CANCELLED'] == 1, col] = flights_df.loc[flights_df['CANCELLED'] == 1, col].fillna(-1)
        flights_df[col] = flights_df[col].fillna(flights_df[col].median())

    cat_cols_to_fill_unknown = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    for col in cat_cols_to_fill_unknown:
        if col in flights_df.columns:
             flights_df[col] = flights_df[col].fillna('Unknown')
    flights_df = flights_df.drop(columns=['CANCELLATION_REASON'], errors='ignore')


    # Define feature sets
    num_feats = ['DISTANCE_KM', 'AIRLINE_7D_MEAN', 'DEP_HOUR', 'DELAY_PER_KM',
                 'DEPARTURE_DELAY', 'ELAPSED_TIME', 'ARRIVAL_DELAY']
    cat_feats = ['DAY_OF_WEEK', 'WEEK_OF_YEAR', 'IS_WEEKEND']
    
    passthrough_label_encode_cols = ['AIRLINE_x', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'TAIL_NUMBER']
    if 'AIRLINE_x' not in flights_df.columns and 'AIRLINE' in flights_df.columns:
        passthrough_label_encode_cols = ['AIRLINE' if item == 'AIRLINE_x' else item for item in passthrough_label_encode_cols]

    # Ensure 'flight_id' is present in the input DataFrame for passthrough
    if 'flight_id' not in flights_df.columns:
        # This should ideally not happen if loaded from DB or CSV where flight_id is present
        # If it might be missing, add it based on index, but this is less robust
        print("Warning: 'flight_id' not found in input to preprocess_data. Adding based on index. Ensure 'flight_id' is correctly loaded.")
        flights_df.insert(0, "flight_id", flights_df.index) # Or range(len(flights_df)) if index is not unique ID

    unchanged_columns = ['flight_id'] 

    all_cols_needed = num_feats + cat_feats + [col for col in passthrough_label_encode_cols if col in flights_df.columns] + unchanged_columns
    # ... (rest of column checking logic) ...
    
    num_pipeline = Pipeline([('impute', SimpleImputer(strategy='median'))])
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=-1)),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    final_passthrough_cols = [col for col in passthrough_label_encode_cols if col in flights_df.columns]
    
    preprocessor_transformers = [
        ('num', num_pipeline, num_feats),
        ('cat', cat_pipeline, cat_feats)
    ]
    if final_passthrough_cols:
        preprocessor_transformers.append(('passthrough_label', 'passthrough', final_passthrough_cols))
    
    # Always include flight_id as a passthrough to retrieve it later
    preprocessor_transformers.append(('passthrough_id', 'passthrough', ['flight_id']))


    ct = ColumnTransformer(preprocessor_transformers, remainder='drop')
    processed_data_placeholder = np.array([])

    if fit_preprocessor:
        # ... (fit logic remains the same) ...
        X_transformed = ct.fit_transform(flights_df) # Pass whole df
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(ct, preprocessor_path)
        print(f"Preprocessor fitted and saved to {preprocessor_path}")
        processed_data_placeholder = X_transformed
    elif preprocessor_path and os.path.exists(preprocessor_path):
        # ... (load logic remains the same) ...
        ct = joblib.load(preprocessor_path)
        X_transformed = ct.transform(flights_df) # Pass whole df
        print("Data transformed using loaded preprocessor.")
        processed_data_placeholder = X_transformed
    # ... (else: error handling or warning) ...

    feature_names = ct.get_feature_names_out()
    processed_df = pd.DataFrame(processed_data_placeholder, columns=feature_names, index=flights_df.index)

    # Extract flight_ids BEFORE any further processing or dropping of rows from processed_df
    # The column name will be 'passthrough_id__flight_id'
    flight_id_col_name_in_processed_df = None
    for col_name in processed_df.columns:
        if 'flight_id' in col_name and 'passthrough_id__' in col_name:
            flight_id_col_name_in_processed_df = col_name
            break
    
    if not flight_id_col_name_in_processed_df:
        raise ValueError("'flight_id' column not found after ColumnTransformer. Check 'passthrough_id' step.")
    
    # Keep flight_ids aligned with the current index of processed_df
    current_flight_ids = processed_df[flight_id_col_name_in_processed_df].copy()


    # TARGET (remains the same)
    y_target = flights_df['ARRIVAL_DELAY'].copy() # Original, non-transformed target from input df
    # Align y_target with processed_df's index (in case any rows were dropped by flights_df processing *before* CT)
    y_target = y_target.loc[processed_df.index]


    label_encoders = {}
    # ... (LabelEncoding logic remains the same) ...
    # Apply Label Encoding to the passthrough columns that were designated for it
    for col_original_name in passthrough_label_encode_cols:
        transformed_col_name = None
        for processed_col_name_iter in processed_df.columns: # Avoid conflict with outer loop var
            if col_original_name in processed_col_name_iter and 'passthrough_label__' in processed_col_name_iter:
                transformed_col_name = processed_col_name_iter
                break
        
        if transformed_col_name and transformed_col_name in processed_df.columns:
            if fit_preprocessor:
                le = LabelEncoder()
                processed_df[transformed_col_name] = le.fit_transform(processed_df[transformed_col_name].astype(str))
                label_encoders[transformed_col_name] = le
            # ... (elif for loading encoders if needed for test_mode, though typically done in make_predictions)

    if fit_preprocessor and label_encoders:
        le_path = os.path.join(os.path.dirname(preprocessor_path), "label_encoders.pkl")
        joblib.dump(label_encoders, le_path)
        print(f"Label encoders saved to {le_path}")


    # Drop rows with NaN in TAIL_NUMBER if it's a passthrough column (example)
    tail_num_col_in_processed = [col for col in processed_df.columns if 'TAIL_NUMBER' in col and 'passthrough_label__' in col]
    if tail_num_col_in_processed:
        # Ensure the column actually exists before trying to dropna on it
        if tail_num_col_in_processed[0] in processed_df.columns:
            original_row_count = len(processed_df)
            processed_df = processed_df.dropna(subset=tail_num_col_in_processed)
            if len(processed_df) < original_row_count:
                print(f"Dropped {original_row_count - len(processed_df)} rows due to NaN in TAIL_NUMBER after transformation.")
                # Re-align y_target and current_flight_ids if rows were dropped
                y_target = y_target.loc[processed_df.index]
                current_flight_ids = current_flight_ids.loc[processed_df.index]


    # Reset index after potential drops for processed_df, y_target, and current_flight_ids
    processed_df.reset_index(drop=True, inplace=True)
    y_target.reset_index(drop=True, inplace=True)
    current_flight_ids.reset_index(drop=True, inplace=True)


    # Data type conversions (remains the same)
    num_transform_cols = [col for col in processed_df.columns if col.startswith('num__')]
    for col in num_transform_cols:
        processed_df[col] = processed_df[col].astype("float32")
    cat_transform_cols = [col for col in processed_df.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]
    for col in cat_transform_cols:
        processed_df[col] = processed_df[col].astype("int32")


    target_col_name_in_processed = 'num__ARRIVAL_DELAY'
    if target_col_name_in_processed not in processed_df.columns:
        raise ValueError(f"Target column {target_col_name_in_processed} not found in processed_df.")

    # X_final should not include the target column or the flight_id column used for joining
    X_final = processed_df.drop(columns=[target_col_name_in_processed, flight_id_col_name_in_processed_df], errors='ignore')
    
    # y_final is the y_target we prepared earlier (original scale, aligned and reset)
    y_final = y_target 
    
    # flight_ids_final is the current_flight_ids (aligned and reset)
    flight_ids_final = current_flight_ids.astype(int) # Ensure it's int

    print(f"Final X shape: {X_final.shape}, Final y shape: {y_final.shape}, Final flight_ids shape: {flight_ids_final.shape}")
    print("Data preprocessing completed.")
    return X_final, y_final, flight_ids_final


if __name__ == '__main__':
    # ... (your existing __main__ for testing preprocess.py) ...
    # Update the call to unpack three values:
    # X_processed, y_target_values, flight_ids_out = preprocess_data(...)
    # print("\nFlight IDs head:")
    # print(flight_ids_out.head())
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "Database/flight_data.db")
    models_dir = os.path.join(project_root, "Models")
    preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")

    from load_data import load_data_from_db
    from feature_engineering import engineer_features

    flights_df_main, airports_df_main, airlines_df_main = load_data_from_db(db_path)
    if not flights_df_main.empty:
        engineered_df_main = engineer_features(flights_df_main, airports_df_main, airlines_df_main)
        
        if 'AIRLINE_x' not in engineered_df_main.columns and 'AIRLINE' in engineered_df_main.columns:
             print("Using 'AIRLINE' for preprocessing as 'AIRLINE_x' is not present.")
        # ...

        X_processed, y_target_values, flight_ids_out = preprocess_data(engineered_df_main, preprocessor_path=preprocessor_save_path, fit_preprocessor=True)
        print("\nProcessed X head:")
        print(X_processed.head())
        print("\nTarget y head:")
        print(y_target_values.head())
        print("\nFlight IDs head:")
        print(flight_ids_out.head())
        # ... (rest of your test logic)