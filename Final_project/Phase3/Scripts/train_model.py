import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
import os
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def smape(y_true, y_pred):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(num / np.maximum(den, 1e-8)) * 100

# train_model now takes X_processed_full, y_target_full, flight_ids_full for the entire dataset
def train_model(X_processed_full, y_target_full, flight_ids_full, model_dir, db_path):
    """
    Splits data, trains the model, saves model artifacts, loss plot,
    and saves the processed X_test, original y_test, and flight_ids_test to the database.
    """
    print("Starting model training and test set saving process...")
    os.makedirs(model_dir, exist_ok=True)

    # Split the full processed data into train/validation and a dedicated test set for DB saving
    # X_processed_full is the output of ColumnTransformer
    # y_target_full is the original scale target
    # flight_ids_full are the corresponding flight IDs
    
    # First split: separate out the test set to be saved
    X_train_val_proc, X_test_proc_to_save, \
    y_train_val_orig, y_test_orig_to_save, \
    ids_train_val, ids_test_to_save = train_test_split(
        X_processed_full, y_target_full, flight_ids_full, 
        test_size=0.05, random_state=42, shuffle=True # 90/10/10
    )

    X_train_proc, X_valid_proc, y_train_orig, y_valid_orig, ids_train, ids_valid = train_test_split(
        X_train_val_proc, y_train_val_orig, ids_train_val,
        test_size=0.05, random_state=42, shuffle=True
    )
    
    print(f"Data split: Train {len(X_train_proc)}, Validation {len(X_valid_proc)}, Test (to DB) {len(X_test_proc_to_save)}")

    # --- Save Processed X_test and original y_test to Database ---
    # X_test_proc_to_save is a DataFrame (output of ColumnTransformer)
    # y_test_orig_to_save is a Series (original arrival delay)
    # ids_test_to_save is a Series (flight_ids)

    # 1. Save y_test_orig_to_save
    df_y_test_to_save = pd.DataFrame({
        'flight_id': ids_test_to_save.values,
        'actual_arrival_delay': y_test_orig_to_save.values
    })
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        # Create saved_y_test table
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS saved_y_test (
            flight_id INTEGER PRIMARY KEY,
            actual_arrival_delay REAL,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id)
        );
        """)
        cur.execute("DELETE FROM saved_y_test;") # Clear previous test actuals
        df_y_test_to_save.to_sql("saved_y_test", con, if_exists="append", index=False)
        print(f"Saved y_test ( {len(df_y_test_to_save)} records) to 'saved_y_test' table.")
        
        # 2. Save X_test_proc_to_save
        # Add flight_id to X_test_proc_to_save DataFrame for saving
        df_X_test_to_save = X_test_proc_to_save.copy()
        df_X_test_to_save.insert(0, 'flight_id', ids_test_to_save.values)

        # Dynamically create table saved_processed_X_test based on df_X_test_to_save columns
        table_name_x_test = "saved_processed_X_test"
        cur.execute(f"DROP TABLE IF EXISTS {table_name_x_test};") # Drop if exists to recreate schema
        df_X_test_to_save.to_sql(table_name_x_test, con, index=False, if_exists="replace") # 'replace' will create based on df
        # Add primary key constraint separately if needed after creation, though flight_id should be unique
        # cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_flight_id_x_test ON {table_name_x_test} (flight_id);")
        print(f"Saved X_test_processed ( {len(df_X_test_to_save)} records, {len(df_X_test_to_save.columns)} columns) to '{table_name_x_test}' table.")
        
        con.commit()
    except Exception as e:
        print(f"Error saving test set (X or y) to database: {e}")
        if con: con.rollback()
    finally:
        if con: con.close()
    # --- End Saving Test Set to DB ---

    # Continue with model training using X_train_proc, y_train_orig, X_valid_proc, y_valid_orig
    numeric_cols = [col for col in X_train_proc.columns if col.startswith('num__')]
    cat_cols_model = [col for col in X_train_proc.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]

    scaler_x = StandardScaler()
    X_train_num_scaled = scaler_x.fit_transform(X_train_proc[numeric_cols])
    X_valid_num_scaled = scaler_x.transform(X_valid_proc[numeric_cols])
    joblib.dump(scaler_x, os.path.join(model_dir, "scaler_x.pkl"))
    print("Numeric feature scaler (scaler_x) saved.")

    cat_inputs_train = {}
    cat_inputs_valid = {}
    cat_input_layers = []
    cat_emb_layers = []

    # For embedding dimensions, use the characteristics of the columns from X_processed_full (before splitting for save)
    # This ensures embedding layers are sized based on the entire potential vocabulary seen during preprocessing.
    temp_full_cat_data_for_embedding_dim = X_processed_full.copy()

    for col in cat_cols_model:
        # Calculate n_cat from the full pre-split processed data to set embedding input_dim
        temp_full_cat_data_for_embedding_dim[col] = temp_full_cat_data_for_embedding_dim[col].fillna(-1).astype("int32")
        max_val_for_col = temp_full_cat_data_for_embedding_dim[col].max()
        n_cat = max_val_for_col + 1
        embed_dim = min(50, (n_cat // 2) + 1)

        inp = layers.Input(shape=(1,), dtype="int32", name=f"{col}_inp")
        cat_input_layers.append(inp)
        emb = layers.Embedding(input_dim=n_cat, output_dim=embed_dim, name=f"{col}_emb")(inp)
        emb = layers.Flatten()(emb)
        cat_emb_layers.append(emb)

        # Use the model's actual train/validation splits for input values
        cat_inputs_train[f"{col}_inp"] = X_train_proc[col].fillna(-1).astype("int32").values
        cat_inputs_valid[f"{col}_inp"] = X_valid_proc[col].fillna(-1).astype("int32").values

    num_inp  = layers.Input(shape=(X_train_num_scaled.shape[1],), name="num_inp")
    if cat_emb_layers: x = layers.concatenate([num_inp] + cat_emb_layers)
    else: x = num_inp
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, name="regression_output")(x)
    model_inputs = [num_inp] + cat_input_layers if cat_input_layers else [num_inp]
    model = models.Model(inputs=model_inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])
    model.summary()

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_orig.values.reshape(-1, 1)).ravel()
    y_valid_scaled = scaler_y.transform(y_valid_orig.values.reshape(-1, 1)).ravel()
    joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))
    print("Target variable scaler (scaler_y) saved.")

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    train_model_inputs = {"num_inp": X_train_num_scaled, **cat_inputs_train}
    valid_model_inputs = {"num_inp": X_valid_num_scaled, **cat_inputs_valid}
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    device_to_use = '/GPU:0' if gpu_devices else '/CPU:0'
    print(f"Using device for training: {device_to_use}")
    with tf.device(device_to_use):
        history = model.fit(
            x=train_model_inputs, y=y_train_scaled,
            validation_data=(valid_model_inputs, y_valid_scaled),
            batch_size=256, epochs=100, callbacks=[early_stop], verbose=1
        )

    model_save_path = os.path.join(model_dir, "flight_delay_model.keras")
    model.save(model_save_path)
    print(f"Trained model saved to {model_save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss During Training'); plt.ylabel('Loss (Scaled MSE)'); plt.xlabel('Epoch')
    plt.legend(loc='upper right'); plt.grid(True)
    plot_save_path = os.path.join(model_dir, "training_loss_plot.png")
    plt.savefig(plot_save_path)
    print(f"Training loss plot saved to {plot_save_path}")

    return model_save_path