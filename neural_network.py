import os
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Keep dataset path independent of the terminal's current directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset.csv"

# MLflow experiment setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("Neural_Network_Classification")

# 1. Load CSV dataset
data = pd.read_csv(DATASET_PATH)
print("Dataset:")
print(data.head())

# 2. Separate input features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training settings (also recorded in MLflow)
epochs = 10
batch_size = 10
learning_rate = 0.001  # Keras Adam default learning rate

with mlflow.start_run(run_name="Keras_Neural_Network"):
    # 5. Create neural network
    model = Sequential([
        Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    # 6. Compile the model
    model.compile(
        optimizer=__import__("tensorflow").keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Log model/training parameters
    mlflow.log_params({
        "model_type": "Sequential Neural Network",
        "input_features": int(X_train.shape[1]),
        "hidden_layer_1_neurons": 16,
        "hidden_layer_2_neurons": 8,
        "output_neurons": 1,
        "hidden_activation": "relu",
        "output_activation": "sigmoid",
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "loss_function": "binary_crossentropy",
        "epochs": epochs,
        "batch_size": batch_size,
        "test_size": 0.2,
        "random_state": 42,
    })

    # 7. Train the model and capture metrics for each epoch
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1,
    )

    for epoch_index, values in enumerate(zip(
        history.history.get("loss", []),
        history.history.get("accuracy", []),
        history.history.get("val_loss", []),
        history.history.get("val_accuracy", []),
    ), start=1):
        train_loss, train_accuracy, val_loss, val_accuracy = values
        mlflow.log_metrics({
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
        }, step=epoch_index)

    # 8. Evaluate on the held-out test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metrics({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    })

    print("\nTest Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # 9. Make predictions
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    print("\nPredicted values:")
    print(predicted_classes)
    print("\nActual values:")
    print(y_test.to_numpy())

    # Save the trained Keras model as an MLflow artifact
    mlflow.tensorflow.log_model(model, artifact_path="neural_network_model")

    # Save the fitted scaler too, so preprocessing can be reused later
    mlflow.sklearn.log_model(scaler, artifact_path="standard_scaler")

print("\nMLflow logging complete.")
print("Run 'mlflow ui' from this project folder and open http://127.0.0.1:5000")
