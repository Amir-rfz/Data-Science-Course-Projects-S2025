import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
import os
import sqlite3
import mlflow 
import mlflow.keras
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlflow.models import infer_signature

def smape(y_true, y_pred):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(num / np.maximum(den, 1e-8)) * 100


def train_model(X_processed_full, y_target_full, flight_ids_full, model_dir, db_path):
    print("Starting model training with MLflow logging and test set saving...")

    mlflow.log_param("full_dataset_rows", X_processed_full.shape[0])
    mlflow.log_param("full_dataset_features_processed", X_processed_full.shape[1])

    X_train_val_proc, X_test_proc_to_save, y_train_val_orig, y_test_orig_to_save, ids_train_val, ids_test_to_save = train_test_split(
        X_processed_full, y_target_full, flight_ids_full, 
        test_size=0.05, random_state=42, shuffle=True 
    ) # 90/5/5

    X_train_proc, X_valid_proc, y_train_orig, y_valid_orig, temp_1, temp_2 = train_test_split(
        X_train_val_proc, y_train_val_orig, ids_train_val,
        test_size=0.05, random_state=42, shuffle=True
    )
    mlflow.log_param("train_set_size", len(X_train_proc))
    mlflow.log_param("validation_set_size", len(X_valid_proc))
    mlflow.log_param("db_test_set_size", len(X_test_proc_to_save))
    
    df_y_test_to_save = pd.DataFrame({'flight_id': ids_test_to_save.values, 'actual_arrival_delay': y_test_orig_to_save.values})
    try:
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS saved_y_test (
            flight_id INTEGER PRIMARY KEY,
            actual_arrival_delay REAL,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id)
        );
        """)
        cur.execute("DELETE FROM saved_y_test;")
        df_y_test_to_save.to_sql("saved_y_test", con, if_exists="append", index=False)
        df_X_test_to_save = X_test_proc_to_save.copy(); df_X_test_to_save.insert(0, 'flight_id', ids_test_to_save.values)
        table_name_x_test = "saved_processed_X_test"; cur.execute(f"DROP TABLE IF EXISTS {table_name_x_test};")
        df_X_test_to_save.to_sql(table_name_x_test, con, index=False, if_exists="replace")
        con.commit(); print("Saved test split to DB.")
    except Exception as e: print(f"Error saving test split to DB: {e}"); mlflow.log_param("db_test_set_save_status", "failed")
    else: mlflow.log_param("db_test_set_save_status", "success")
    finally:
        if con: con.close()


    batch_size_param = 256
    epochs_param = 100
    optimizer_param = "adam"
    early_stopping_patience = 20

    mlflow.log_params({
        "batch_size": batch_size_param,
        "epochs": epochs_param,
        "optimizer": optimizer_param,
        "early_stopping_patience": early_stopping_patience,
        "target_variable": "ARRIVAL_DELAY"
    })

    numeric_cols = [col for col in X_train_proc.columns if col.startswith('num__')]
    scaler_x = StandardScaler()
    X_train_num_scaled = scaler_x.fit_transform(X_train_proc[numeric_cols])
    X_valid_num_scaled = scaler_x.transform(X_valid_proc[numeric_cols])
    scaler_x_path = os.path.join(model_dir, "scaler_x.pkl")
    joblib.dump(scaler_x, scaler_x_path)
    mlflow.log_artifact(scaler_x_path, artifact_path="scalers")
    print("Numeric feature scaler (scaler_x) saved and logged.")


    cat_cols_model = [col for col in X_train_proc.columns if col.startswith('cat__') or col.startswith('passthrough_label__')]
    cat_inputs_train = {}
    cat_inputs_valid = {}
    cat_input_layers = []
    cat_emb_layers = []
    embedding_details = {} 

    temp_full_cat_data_for_embedding_dim = X_processed_full.copy()
    for col in cat_cols_model:
        temp_full_cat_data_for_embedding_dim[col] = temp_full_cat_data_for_embedding_dim[col].fillna(-1).astype("int32")
        max_val_for_col = temp_full_cat_data_for_embedding_dim[col].max()
        n_cat = max_val_for_col + 1
        embed_dim = min(50, (n_cat // 2) + 1)
        embedding_details[col] = {"input_dim": n_cat, "output_dim": embed_dim}
        inp = layers.Input(shape=(1,), dtype="int32", name=f"{col}_inp"); cat_input_layers.append(inp)
        emb = layers.Embedding(input_dim=n_cat, output_dim=embed_dim, name=f"{col}_emb")(inp)
        emb = layers.Flatten()(emb); cat_emb_layers.append(emb)
        cat_inputs_train[f"{col}_inp"] = X_train_proc[col].fillna(-1).astype("int32").values
        cat_inputs_valid[f"{col}_inp"] = X_valid_proc[col].fillna(-1).astype("int32").values
    mlflow.log_text(json.dumps(embedding_details, indent=2), "embedding_details.json")


    num_inp  = layers.Input(shape=(X_train_num_scaled.shape[1],), name="num_inp")
    if cat_emb_layers: 
        x = layers.concatenate([num_inp] + cat_emb_layers)
    else: 
        x = num_inp
    dense_layers_config = [256, 128, 64, 32] 
    mlflow.log_param("dense_layers", str(dense_layers_config))

    x = layers.Dense(dense_layers_config[0], activation="relu")(x)
    x = layers.Dense(dense_layers_config[1], activation="relu")(x)
    x = layers.Dense(dense_layers_config[2], activation="relu")(x)
    x = layers.Dense(dense_layers_config[3], activation="relu")(x)
    out = layers.Dense(1, name="regression_output")(x)
    model_inputs = [num_inp] + cat_input_layers if cat_input_layers else [num_inp]
    model = models.Model(inputs=model_inputs, outputs=out)
    model.compile(optimizer=optimizer_param, loss="mse", metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])
    model.summary() 

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_orig.values.reshape(-1, 1)).ravel()
    y_valid_scaled = scaler_y.transform(y_valid_orig.values.reshape(-1, 1)).ravel()
    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
    joblib.dump(scaler_y, scaler_y_path)
    mlflow.log_artifact(scaler_y_path, artifact_path="scalers") 
    print("Target variable scaler (scaler_y) saved and logged.")
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)

    gpu_devices = tf.config.list_physical_devices('GPU'); device_to_use = '/GPU:0' if gpu_devices else '/CPU:0'
    print(f"Using device for training: {device_to_use}")

    with tf.device(device_to_use):
        history = model.fit(
            x={"num_inp": X_train_num_scaled, **cat_inputs_train},
            y=y_train_scaled,
            validation_data=({"num_inp": X_valid_num_scaled, **cat_inputs_valid}, y_valid_scaled),
            batch_size=batch_size_param,
            epochs=epochs_param,
            callbacks=[early_stop], 
            verbose=1
        )

    val_loss_best, val_mae_best, val_rmse_best = model.evaluate(
        {"num_inp": X_valid_num_scaled, **cat_inputs_valid}, y_valid_scaled, verbose=0
    )
    mlflow.log_metrics({
        "best_val_loss_mse": val_loss_best, 
        "best_val_mae": val_mae_best,
        "best_val_rmse": val_rmse_best
    })
    print(f"Best validation metrics: Loss (MSE)={val_loss_best:.4f}, MAE={val_mae_best:.4f}, RMSE={val_rmse_best:.4f}")

    model_save_path = os.path.join(model_dir, "flight_delay_model.keras")
    model.save(model_save_path)    
    
    mlflow.keras.log_model(
        model,
        artifact_path="keras_model",
        registered_model_name="FlightDelayKerasModel",
    )

    print(f"Trained Keras model logged to MLflow.")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss (Scaled MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plot_save_path = os.path.join(model_dir, "training_loss_plot.png") 
    plt.savefig(plot_save_path)
    mlflow.log_artifact(plot_save_path, artifact_path="plots")
    print(f"Training loss plot saved to {plot_save_path} and logged.")

    return "MLflow_Logged_Model" 