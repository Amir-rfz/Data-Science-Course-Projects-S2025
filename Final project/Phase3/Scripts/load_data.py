import sqlite3
import pandas as pd

def load_data_from_db(db_path):
    con = sqlite3.connect(db_path)
    try:
        flights = pd.read_sql_query("SELECT * FROM flights", con)
        airports = pd.read_sql_query("SELECT * FROM airports", con)
        airlines = pd.read_sql_query("SELECT * FROM airlines", con)
        print("Data loaded successfully from database.")
        return flights, airports, airlines
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 
    finally:
        con.close()