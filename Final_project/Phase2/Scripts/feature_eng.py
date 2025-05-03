from load_data import load_data 
import pandas as pd

def feature_eng():
    flights, airports, airlines = load_data()
    
    #add latitude and longitude to calculate distance 
    airp_cols = ["IATA_CODE", "LATITUDE", "LONGITUDE"]
    orig = airports[airp_cols].rename(columns=lambda c: "ORIG_" + c if c!="IATA_CODE" else "ORIGIN_AIRPORT")
    dest = airports[airp_cols].rename(columns=lambda c: "DEST_" + c if c!="IATA_CODE" else "DESTINATION_AIRPORT")

    flights = (flights
            .merge(orig,  on="ORIGIN_AIRPORT",      how="left")
            .merge(dest,  on="DESTINATION_AIRPORT", how="left")
            .merge(airlines, on="AIRLINE",           how="left"))

    flights = flights.drop(columns=['IATA_CODE'], errors='ignore')

    # Add date column:
    flights['FLIGHT_DATE'] = pd.to_datetime(
        flights[['A_YEAR','A_MONTH','A_DAY']].rename(
            columns={'A_YEAR':'year','A_MONTH':'month','A_DAY':'day'}),
        format="%Y-%m-%d"
    )

    flights['DAY_OF_YEAR'] = flights['FLIGHT_DATE'].dt.dayofyear
    flights['WEEK_OF_YEAR'] = flights['FLIGHT_DATE'].dt.isocalendar().week
    flights['QUARTER'] = flights['FLIGHT_DATE'].dt.quarter
    flights['IS_WEEKEND'] = flights['FLIGHT_DATE'].dt.weekday >= 5  

    flights['DEP_HOUR'] = (flights['SCHEDULED_DEPARTURE'] // 100).astype(int)
    bins = [0, 6, 12, 18, 24]
    labels = ['early_morning','morning','afternoon','evening']
    flights['DEP_TIME_BUCKET'] = pd.cut(flights['DEP_HOUR'], bins=bins, labels=labels, right=False)

    import pyproj
    import numpy as np

    # Add distance column:
    proj = pyproj.Proj(proj='latlong', datum='WGS84')
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * 6371 * np.arcsin(np.sqrt(a))  

    flights['DISTANCE_KM'] = haversine(
        flights['ORIG_LATITUDE'], flights['ORIG_LONGITUDE'],
        flights['DEST_LATITUDE'], flights['DEST_LONGITUDE']
    )

    # Calculating 7 previous flights avg delay:
    flights = flights.sort_values(["AIRLINE","FLIGHT_DATE"])
    flights["ARR_DELAY_FILLED"] = flights["ARRIVAL_DELAY"].fillna(0) 

    flights["AIRLINE_7D_MEAN"] = (
        flights
        .groupby("AIRLINE")["ARR_DELAY_FILLED"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0,drop=True)
    )

    flights.drop(columns="ARR_DELAY_FILLED", inplace=True)

    # Delay per km column:
    flights['DELAY_PER_KM'] = flights['ARRIVAL_DELAY'] / flights['DISTANCE_KM']

    # Drop unnecessary columns:
    drop_cols = ['flight_id','A_YEAR','A_MONTH','A_DAY',
             'ORIG_LATITUDE','ORIG_LONGITUDE','DEST_LATITUDE','DEST_LONGITUDE', 'TAIL_NUMBER']
    flights = flights.drop(columns=drop_cols, errors='ignore')

    return flights

if __name__ == "__main__":
    feature_eng_df = feature_eng()
    feature_eng_df.to_csv("feature_eng.csv", index=False)