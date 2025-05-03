import pandas as pd
from database_connection import connect_to_db

def load_data():
    con = connect_to_db()
    flights = pd.read_sql_query("SELECT * FROM flights", con)
    airports = pd.read_sql_query("SELECT * FROM airports", con)
    airlines = pd.read_sql_query("SELECT * FROM airlines", con)
    con.commit()
    con.close()
    return flights, airports, airlines