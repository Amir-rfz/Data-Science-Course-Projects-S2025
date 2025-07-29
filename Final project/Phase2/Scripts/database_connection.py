import sqlite3
db_path = "./Database/flight_data.db"
def connect_to_db():
    con = sqlite3.connect(db_path)
    return con