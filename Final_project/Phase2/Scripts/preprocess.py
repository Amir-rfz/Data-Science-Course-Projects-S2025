from feature_eng import feature_eng
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

def preprocess():
    flights = feature_eng()
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
