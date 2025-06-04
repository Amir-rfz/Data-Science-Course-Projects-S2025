# Scripts/load_data.py
import sqlite3
import pandas as pd

def load_data_from_db(db_path):
    """
    Loads data from the SQLite database into pandas DataFrames.
    """
    con = sqlite3.connect(db_path)
    try:
        flights = pd.read_sql_query("SELECT * FROM flights", con)
        airports = pd.read_sql_query("SELECT * FROM airports", con)
        airlines = pd.read_sql_query("SELECT * FROM airlines", con)
        print("Data loaded successfully from database.")
        return flights, airports, airlines
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty DataFrames on error
    finally:
        con.close()

if __name__ == '__main__':
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "Database/flight_data.db")
    
    # Example usage:
    flights_df, airports_df, airlines_df = load_data_from_db(db_path)
    if not flights_df.empty:
        print("\nFlights DataFrame head:")
        print(flights_df.head())
    if not airports_df.empty:
        print("\nAirports DataFrame head:")
        print(airports_df.head())
    if not airlines_df.empty:
        print("\nAirlines DataFrame head:")
        print(airlines_df.head())