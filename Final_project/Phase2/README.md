# Data Pipeline

## Overview
This project processes and transforms data in a multi-step pipeline.

## Steps
1. `import_to_db.py`: Imports the data into SQLite.
2. `database_connection.py`: Connect to database.
3. `load_data.py`: Loads and inspects the raw data.
4. `preprocess.py`: Cleans and standardizes the data.
5. `feature_eng.py`: Adds derived features.

## How to Run
Execute the entire pipeline:
    ```bash
    python pipeline.py
    ```

## Output Files
- `feature_eng.csv`: values haven't been normalized but new features are added
- `processed.csv`: Dataset with new columns and normalized values
