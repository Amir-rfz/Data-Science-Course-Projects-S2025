# Scripts/create_database.py
import sqlite3
import pandas as pd
import os

def build_database(db_path, flights_csv, airports_csv, airlines_csv):
    """
    Creates the SQLite database, tables, and populates them with initial data.
    """
    # Ensure the Database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    airports_df = pd.read_csv(airports_csv)
    airlines_df = pd.read_csv(airlines_csv)
    flights_df = pd.read_csv(flights_csv)

    # Basic cleaning and selection
    airports_df = airports_df[["IATA_CODE", "AIRPORT", "CITY", "A_STATE",
                               "COUNTRY", "LATITUDE", "LONGITUDE"]].drop_duplicates()
    airlines_df = airlines_df[["IATA_CODE", "AIRLINE"]].drop_duplicates()
    flights_df = flights_df[["A_YEAR", "A_MONTH", "A_DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
                             "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE",
                             "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "SCHEDULED_TIME",
                             "ELAPSED_TIME", "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN",
                             "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY", "DIVERTED", "CANCELLED",
                             "CANCELLATION_REASON", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
                             "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]].drop_duplicates()
    flights_df.insert(0, "flight_id", range(1, len(flights_df) + 1))

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    cur = con.cursor()

    for tbl in ("flights", "airports", "airlines"):
        cur.executescript(f"DROP TABLE IF EXISTS {tbl};")

    cur.executescript("""
    CREATE TABLE airports(
        IATA_CODE TEXT PRIMARY KEY,
        AIRPORT TEXT,
        CITY TEXT,
        A_STATE TEXT,
        COUNTRY TEXT,
        LATITUDE REAL,
        LONGITUDE REAL
    );
    """)

    cur.executescript("""
    CREATE TABLE airlines(
        IATA_CODE TEXT PRIMARY KEY,
        AIRLINE TEXT
    );
    """)

    cur.executescript("""
    CREATE TABLE flights(
        flight_id INT PRIMARY KEY,
        A_YEAR INT,
        A_MONTH INT,
        A_DAY INT,
        DAY_OF_WEEK INT,
        AIRLINE TEXT,
        FLIGHT_NUMBER INT,
        TAIL_NUMBER TEXT,
        ORIGIN_AIRPORT TEXT,
        DESTINATION_AIRPORT TEXT,
        SCHEDULED_DEPARTURE INT,
        DEPARTURE_TIME INT,
        DEPARTURE_DELAY INT,
        TAXI_OUT INT,
        WHEELS_OFF INT,
        SCHEDULED_TIME INT,
        ELAPSED_TIME INT,
        AIR_TIME INT,
        DISTANCE INT,
        WHEELS_ON INT,
        TAXI_IN INT,
        SCHEDULED_ARRIVAL INT,
        ARRIVAL_TIME INT,
        ARRIVAL_DELAY INT,
        DIVERTED INT,
        CANCELLED INT,
        CANCELLATION_REASON TEXT,
        AIR_SYSTEM_DELAY INT,
        SECURITY_DELAY INT,
        AIRLINE_DELAY INT,
        LATE_AIRCRAFT_DELAY INT,
        WEATHER_DELAY INT,
        FOREIGN KEY (ORIGIN_AIRPORT) REFERENCES airports(IATA_CODE),
        FOREIGN KEY (DESTINATION_AIRPORT) REFERENCES airports(IATA_CODE),
        FOREIGN KEY (AIRLINE) REFERENCES airlines(IATA_CODE)
    );
    """)
    con.commit()

    try:
        airports_df.to_sql("airports", con, if_exists="append", index=False)
        airlines_df.to_sql("airlines", con, if_exists="append", index=False)
        flights_df.to_sql("flights", con, if_exists="append", index=False)
        con.commit()
        print(f"SQLite database built and populated at {db_path}")
    except Exception as e:
        print(f"Error populating tables: {e}")
        con.rollback()
    finally:
        con.close()

if __name__ == '__main__':
    # Define paths relative to the Scripts directory or use absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get project root
    db_path = os.path.join(project_root, "Database/flight_data.db")
    flights_csv = os.path.join(project_root, "Dataset/flights.csv")
    airports_csv = os.path.join(project_root, "Dataset/airports.csv")
    airlines_csv = os.path.join(project_root, "Dataset/airlines.csv")

    build_database(db_path, flights_csv, airports_csv, airlines_csv)