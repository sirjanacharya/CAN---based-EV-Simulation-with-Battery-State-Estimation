import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# === DATA LOADING AND PREPROCESSING ===
def load_and_preprocess_data(csv_file, sequence_length=64, test_size=0.2):
    df = pd.read_csv(csv_file, usecols=[0, 1, 2, 3])  # Voltage, Current, Time, SOC
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    df.iloc[:, [0, 1, 2]] = scaler_x.fit_transform(df.iloc[:, [0, 1, 2]])
    df.iloc[:, 3] = df.iloc[:, 3] = scaler_y.fit_transform(df.iloc[:, 3].values.reshape(-1, 1)).flatten()
    

    data = df.iloc[:, [0, 1, 2]].values  # Features (Voltage, Current, Time)
    labels = df.iloc[:, 3].values  # SOC target

    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(labels[i])

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y

# === MODEL DEFINITION ===
class LSTM_Attention_SOC(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2, num_heads=4, dropout=0.2):
        super(LSTM_Attention_SOC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, 
                                                     dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return out

# === TRAINING FUNCTION WITH LOSS TRACKING ===
def train_lstm_attention(X_train, y_train,
                         input_size=3, hidden_size=32, num_layers=2, num_heads=4, 
                         num_epochs=30, batch_size=16, learning_rate=0.00015):
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTM_Attention_SOC(input_size, hidden_size, num_layers, num_heads).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

\

    print("\n🔄 Training LSTM-Attention Model on", device, "...\n")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)


        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}")

    print("\n✅ Training Complete!\n")


    return model
    # Main training execution
if __name__ == "__main__":
    csv_file = "/u/student/2024/cs24resch11014/sirjan/merged_data.csv"

    X_train, X_test, y_train, y_test, scaler_x, scaler_y = load_and_preprocess_data(csv_file)

    model = train_lstm_attention(X_train, y_train)

    # Save model
    model_save_path = "lstm_attn_soc_acs_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 Model saved to {model_save_path}\n")

    # Save scaler_x if needed
    scaler_save_path = "scaler_x_acs.pkl"
    pd.to_pickle(scaler_x, scaler_save_path)
    print(f"💾 Scaler saved to {scaler_save_path}\n")

        # Save scaler_y if needed
    scaler_save_path = "scaler_y_acs.pkl"
    pd.to_pickle(scaler_y, scaler_save_path)
    print(f"💾 Scaler saved to {scaler_save_path}\n")
