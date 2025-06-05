import os
import pandas as pd
import numpy as np
from load_data import load_data_from_db

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 
    return c * r

def engineer_features(flights_df, airports_df, airlines_df):
    print("Starting feature engineering...")
    airp_cols = ["IATA_CODE", "LATITUDE", "LONGITUDE"]
    orig = airports_df[airp_cols].rename(columns=lambda c: "ORIG_" + c if c != "IATA_CODE" else "ORIGIN_AIRPORT")
    dest = airports_df[airp_cols].rename(columns=lambda c: "DEST_" + c if c != "IATA_CODE" else "DESTINATION_AIRPORT")

    flights = (flights_df
               .merge(orig, on="ORIGIN_AIRPORT", how="left")
               .merge(dest, on="DESTINATION_AIRPORT", how="left")
               .merge(airlines_df, left_on="AIRLINE", right_on="IATA_CODE", how="left", suffixes=('', '_airline_code'))) 

    if 'IATA_CODE_airline_code' in flights.columns:
        flights = flights.drop(columns=['IATA_CODE_airline_code'])
    if 'IATA_CODE' in flights.columns and 'AIRLINE_x' not in flights.columns : 
        if len(flights.AIRLINE.unique()) == len(flights.IATA_CODE.unique()): 
            pass 

    flights['FLIGHT_DATE'] = pd.to_datetime(
        flights[['A_YEAR', 'A_MONTH', 'A_DAY']].rename(
            columns={'A_YEAR': 'year', 'A_MONTH': 'month', 'A_DAY': 'day'}),
        format="%Y-%m-%d"
    )

    flights['DAY_OF_YEAR'] = flights['FLIGHT_DATE'].dt.dayofyear
    flights['WEEK_OF_YEAR'] = flights['FLIGHT_DATE'].dt.isocalendar().week.astype(int) 
    flights['QUARTER'] = flights['FLIGHT_DATE'].dt.quarter
    flights['IS_WEEKEND'] = (flights['FLIGHT_DATE'].dt.weekday >= 5).astype(int) 

    flights['DEP_HOUR'] = (flights['SCHEDULED_DEPARTURE'] // 100).astype(int)
    bins = [0, 6, 12, 18, 24]
    labels = ['early_morning', 'morning', 'afternoon', 'evening']
    flights['DEP_TIME_BUCKET'] = pd.cut(flights['DEP_HOUR'], bins=bins, labels=labels, right=False)

    flights['DISTANCE_KM'] = haversine(
        flights['ORIG_LATITUDE'].values, flights['ORIG_LONGITUDE'].values, 
        flights['DEST_LATITUDE'].values, flights['DEST_LONGITUDE'].values
    )
    flights['DISTANCE_KM'] = flights['DISTANCE_KM'].astype(float)

    flights["ARR_DELAY_FILLED"] = flights["ARRIVAL_DELAY"].fillna(0)
    flights = flights.sort_values(["AIRLINE", "FLIGHT_DATE"])
    flights["AIRLINE_7D_MEAN"] = (
        flights
        .groupby("AIRLINE")["ARR_DELAY_FILLED"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    flights.drop(columns="ARR_DELAY_FILLED", inplace=True)
    flights["AIRLINE_7D_MEAN"] = flights["AIRLINE_7D_MEAN"].astype(float) 

    flights['DELAY_PER_KM'] = (flights['ARRIVAL_DELAY'] / flights['DISTANCE_KM'].replace(0, np.nan)).fillna(0)
    flights['DELAY_PER_KM'] = flights['DELAY_PER_KM'].astype(float) 
    
    print("Feature engineering completed.")
    return flights

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "Database/flight_data.db")
    flights_df, airports_df, airlines_df = load_data_from_db(db_path)
    engineered_flights_df = engineer_features(flights_df, airports_df, airlines_df)