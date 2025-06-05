import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_data(flights_df, preprocessor_path=None, fit_preprocessor=False):

    print("Starting data preprocessing...")

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

    num_feats = ['DISTANCE_KM', 'AIRLINE_7D_MEAN', 'DEP_HOUR', 'DELAY_PER_KM',
                 'DEPARTURE_DELAY', 'ELAPSED_TIME', 'ARRIVAL_DELAY']
    cat_feats = ['DAY_OF_WEEK', 'WEEK_OF_YEAR', 'IS_WEEKEND']
    
    passthrough_label_encode_cols = ['AIRLINE_x', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'TAIL_NUMBER']
    if 'AIRLINE_x' not in flights_df.columns and 'AIRLINE' in flights_df.columns:
        passthrough_label_encode_cols = ['AIRLINE' if item == 'AIRLINE_x' else item for item in passthrough_label_encode_cols]

    if 'flight_id' not in flights_df.columns:
        flights_df.insert(0, "flight_id", flights_df.index) 

    unchanged_columns = ['flight_id'] 
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
    
    preprocessor_transformers.append(('passthrough_id', 'passthrough', unchanged_columns))


    ct = ColumnTransformer(preprocessor_transformers, remainder='drop')
    processed_data_placeholder = np.array([])

    if fit_preprocessor:
        X_transformed = ct.fit_transform(flights_df) 
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(ct, preprocessor_path)
        print(f"Preprocessor fitted and saved to {preprocessor_path}")
        processed_data_placeholder = X_transformed
    elif preprocessor_path and os.path.exists(preprocessor_path):
        ct = joblib.load(preprocessor_path)
        X_transformed = ct.transform(flights_df) 
        print("Data transformed using loaded preprocessor.")
        processed_data_placeholder = X_transformed

    feature_names = ct.get_feature_names_out()
    processed_df = pd.DataFrame(processed_data_placeholder, columns=feature_names, index=flights_df.index)

    flight_id_col_name_in_processed_df = None
    for col_name in processed_df.columns:
        if 'flight_id' in col_name and 'passthrough_id__' in col_name:
            flight_id_col_name_in_processed_df = col_name
            break
    
    current_flight_ids = processed_df[flight_id_col_name_in_processed_df].copy()


    y_target = flights_df['ARRIVAL_DELAY'].copy() 
    y_target = y_target.loc[processed_df.index]


    label_encoders = {}
    for col_original_name in passthrough_label_encode_cols:
        transformed_col_name = None
        for processed_col_name_iter in processed_df.columns:
            if col_original_name in processed_col_name_iter and 'passthrough_label__' in processed_col_name_iter:
                transformed_col_name = processed_col_name_iter
                break
        
        if transformed_col_name and transformed_col_name in processed_df.columns:
            if fit_preprocessor:
                le = LabelEncoder()
                processed_df[transformed_col_name] = le.fit_transform(processed_df[transformed_col_name].astype(str))
                label_encoders[transformed_col_name] = le

    if fit_preprocessor and label_encoders:
        le_path = os.path.join(os.path.dirname(preprocessor_path), "label_encoders.pkl")
        joblib.dump(label_encoders, le_path)
        print(f"Label encoders saved to {le_path}")


    tail_num_col_in_processed = [col for col in processed_df.columns if 'TAIL_NUMBER' in col and 'passthrough_label__' in col]
    if tail_num_col_in_processed:
        if tail_num_col_in_processed[0] in processed_df.columns:
            original_row_count = len(processed_df)
            processed_df = processed_df.dropna(subset=tail_num_col_in_processed)
            if len(processed_df) < original_row_count:
                print(f"Dropped {original_row_count - len(processed_df)} rows due to NaN in TAIL_NUMBER after transformation.")
                y_target = y_target.loc[processed_df.index]
                current_flight_ids = current_flight_ids.loc[processed_df.index]


    processed_df.reset_index(drop=True, inplace=True)
    y_target.reset_index(drop=True, inplace=True)
    current_flight_ids.reset_index(drop=True, inplace=True)


    num_transform_cols = [col for col in processed_df.columns if col.startswith('num__')]
    for col in num_transform_cols:
        processed_df[col] = processed_df[col].astype("float32")
    cat_transform_cols = [col for col in processed_df.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]
    for col in cat_transform_cols:
        processed_df[col] = processed_df[col].astype("int32")


    target_col_name_in_processed = 'num__ARRIVAL_DELAY'
    if target_col_name_in_processed not in processed_df.columns:
        raise ValueError(f"Target column {target_col_name_in_processed} not found in processed_df.")

    X_final = processed_df.drop(columns=[target_col_name_in_processed, flight_id_col_name_in_processed_df], errors='ignore')
    y_final = y_target 
    flight_ids_final = current_flight_ids.astype(int) 

    print(f"Final X shape: {X_final.shape}, Final y shape: {y_final.shape}, Final flight_ids shape: {flight_ids_final.shape}")
    print("Data preprocessing completed.")
    return X_final, y_final, flight_ids_final