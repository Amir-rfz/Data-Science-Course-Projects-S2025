from feature_eng import feature_eng
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

def preprocess():
    flights = feature_eng()
    
    delay_cols = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
    flights[delay_cols] = flights[delay_cols].fillna(0)

    delay_basic_cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY']
    flights[delay_basic_cols] = flights[delay_basic_cols].fillna(0)

    time_cols = ['DEPARTURE_TIME', 'ARRIVAL_TIME', 'WHEELS_ON', 'WHEELS_OFF', 
                'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME']

    for col in time_cols:
        flights.loc[flights['CANCELLED'] == 1, col] = flights.loc[flights['CANCELLED'] == 1, col].fillna(-1)

    for col in time_cols:
        flights[col] = flights[col].fillna(flights[col].median())

    cat_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    flights[cat_cols] = flights[cat_cols].fillna('Unknown')

    flights = flights.drop(columns=['CANCELLATION_REASON'], errors='ignore')

    num_feats = ['DISTANCE_KM', 'AIRLINE_7D_MEAN','DEP_HOUR','DELAY_PER_KM','DEPARTURE_DELAY', 'SCHEDULED_DEPARTURE',
                'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN', 'AIR_TIME', 'ELAPSED_TIME']

    cat_feats = ['DAY_OF_WEEK', 'WEEK_OF_YEAR', 'IS_WEEKEND']
    unchanged_columns = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_DATE']

    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=-1)),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_feats),
        ('cat', cat_pipeline, cat_feats),
        ('passthrough', 'passthrough', unchanged_columns)
    ])

    X = preprocessor.fit_transform(flights)
    y = (flights['ARRIVAL_DELAY'] > 15).astype(int)

    columns = preprocessor.get_feature_names_out()
    processed = pd.DataFrame(X, 
                            index=flights.index,
                            columns = (
                                columns
                            ))
    processed['TARGET_LATE'] = y

    return processed


if __name__ == "__main__":
    processed = preprocess()
    processed.to_csv("processed.csv", index=False)
