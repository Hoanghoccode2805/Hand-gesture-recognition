from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
try:
    from src.preprocessing import Preprocessor
except ModuleNotFoundError:
    from preprocessing import Preprocessor

class GestureMLP(nn.Module):                                # khởi tạo class con từ class cha là nn.Module
    def __init__(self, input_size = 63, num_classes = 4):
        super().__init__()          # gọi hàm khởi tạo (__init__) của parent class từ bên trong child class.
        self.network = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,num_classes)
        )

    def forward(self, x):
        return self.network(x)

def main():
    csv_path = Path("data/gesture_data.csv")
    df = pd.read_csv(csv_path)

    X_train, X_val, y_train, y_val = Preprocessor.load_and_split_data()

    # 1. Convert to Tensors (X must be float32, y must be long for CrossEntropyLoss)
    X_train_tensor = torch.tensor(X_train, dtype= torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype= torch.long)
    X_val_tensor = torch.tensor(X_val, dtype= torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype= torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size= 32,
        shuffle= True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size= 32,
        shuffle= False
    )

    # 2. INITIALIZE MODEL, LOSS, AND OPTIMIZER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureMLP(input_size=63, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # 3. TRAINING LOOP (50 EPOCHS)
    epochs = 70
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()                   # 1. Xóa gradient cũ
            outputs = model(X_batch)                # 2. Forward pass (dự đoán)
            loss = criterion(outputs, y_batch)      # 3. Tính sai số
            loss.backward()                         # 4. Backpropagation (tính đạo hàm)
            optimizer.step()                        # 5. Cập nhật trọng số

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # 4. Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                # Lấy lớp có điểm số cao nhất làm nhãn dự đoán
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset) * 100

        if epoch % 5 == 0 or epoch == 1:             # In kết quả mỗi 5 epochs để dễ theo dõi
            print(
                f"Epoch [{epoch:02d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )


    model_path = Path("models/mlp.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n Model successfully saved at: {model_path}")


if __name__ == "__main__":
    main()