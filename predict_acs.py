# predict_lstm_model.py

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from train_acs import LSTM_Attention_SOC, load_and_preprocess_data

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

# Predict function
def predict_lstm(model, X_test, batch_size=16):
    """
    Predicts SOC values batch-wise to avoid CUDA OOM errors.
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            preds.append(y_pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # Combine all batch predictions
    return preds

# Main prediction execution
if __name__ == "__main__":
    # CSV file to predict
    csv_file = "/u/student/2024/cs24resch11014/sirjan/sample1.csv"

    # Load the data
    df = pd.read_csv(csv_file, usecols=[0, 1, 2, 3])  # Voltage, Current, Time, SOC
    X_data = df.iloc[:, [0, 1, 2]].values  # Features
    y_true = df.iloc[:, 3].values          # True SOC


    # Load scalers
    scaler_x = pd.read_pickle("scaler_x_acs.pkl")
    scaler_y = pd.read_pickle("scaler_y_acs.pkl")

    # Normalize inputs
    X_data_df = pd.DataFrame(X_data, columns=["Voltage_measured", "Current_measured", "Time"])
    X_scaled = scaler_x.transform(X_data_df)
    print(X_scaled)

    # Normalize labels (true SOC)
    y_true_scaled = scaler_y.transform(y_true.reshape(-1, 1)).flatten()
    print(y_true_scaled)

    # Create sequences
    sequence_length = 64  # Same as used during training
    X_sequences = np.array([X_scaled[i:i + sequence_length] for i in range(len(X_scaled) - sequence_length)])
    y_sequences = y_true_scaled[sequence_length:]  # Shifted true SOC labels

    # Load trained model
    input_size = 3
    hidden_size = 32
    num_layers = 2
    num_heads = 4

    model = LSTM_Attention_SOC(input_size, hidden_size, num_layers, num_heads).to(device)
    model.load_state_dict(torch.load("lstm_attn_soc_acs_model.pth", map_location=device))
    model.eval()

    # Predict
    y_pred_scaled = predict_lstm(model, X_sequences, batch_size=32)

    # Inverse transform the predictions and ground truth
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_sequences.reshape(-1, 1)).flatten()

    # # Evaluation
    # mse = np.mean((y_true - y_pred) ** 2)
    # mae = np.mean(np.abs(y_true - y_pred))
    # rmse = np.sqrt(mse)

    # print("\n📊 Evaluation Metrics on Sample.csv:")
    # print(f"📉 Mean Squared Error (MSE): {mse:.6f}")
    # print(f"📉 Mean Absolute Error (MAE): {mae:.6f}")
    # print(f"📉 Root Mean Squared Error (RMSE): {rmse:.6f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True SOC", color="blue", alpha=0.7)
    plt.plot(y_pred, label="Predicted SOC", color="red", alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("SOC")
    plt.title("True vs Predicted SOC (Sample.csv)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # Save plot
    plt.savefig("sample_soc_prediction_acs1.png")
    print("📈 Saved plot as 'sample_soc_acs_prediction1.png'")
    plt.show()

