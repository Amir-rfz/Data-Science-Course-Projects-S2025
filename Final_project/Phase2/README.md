# Data Pipeline

## Overview
This project processes and transforms data in a multi-step pipeline.

## Steps
1. `import_to_dp.py`: Seeds the database with raw data.
2. `database_connection.py`: Seeds the database with raw data.
3. `load_data.py`: Loads and inspects the raw data.
4. `preprocess.py`: Cleans and standardizes the data.
5. `feature_eng.py`: Adds derived features.

## How to Run
Execute the entire pipeline:
    ```bash
    python pipeline.py
    ```

## Output Files
- `feature_eng.csv`: values havent been normalized but new features are added
- `processed.csv`: Dataset with new columns and normalized values
