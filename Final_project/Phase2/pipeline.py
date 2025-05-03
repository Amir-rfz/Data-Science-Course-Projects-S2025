import subprocess

subprocess.run(["python", "Scripts/import_to_db.py"])
subprocess.run(["python", "Scripts/database_connection.py"])
subprocess.run(["python", "Scripts/load_data.py"])
subprocess.run(["python", "Scripts/preprocess.py"])
subprocess.run(["python", "Scripts/feature_eng.py"])
