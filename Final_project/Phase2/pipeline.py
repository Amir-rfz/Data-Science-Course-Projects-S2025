import subprocess
import shutil, pathlib

subprocess.run(["python", "Scripts/import_to_db.py"])
subprocess.run(["python", "Scripts/database_connection.py"])
subprocess.run(["python", "Scripts/load_data.py"])
subprocess.run(["python", "Scripts/preprocess.py"])
subprocess.run(["python", "Scripts/feature_eng.py"])

root = pathlib.Path(__file__).resolve().parent
phase2_dir = root                             
dest_dir = phase2_dir / "final_outputs"
dest_dir.mkdir(exist_ok=True)

for fname in ("feature_eng.csv", "processed.csv"):
    src = phase2_dir / fname
    shutil.move(src, dest_dir / fname)
