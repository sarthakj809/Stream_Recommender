import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import pickle

# ------------- Data preprocessing -------------
df = pd.read_csv("data/student-scores-with-stream.csv")

features = [
    'math_score', 'physics_score', 'biology_score', 'chemistry_score',
    'english_score', 'history_score', 'geography_score'
]

X_numeric = df[features].copy()

career_ohe = OneHotEncoder(sparse_output=False)
career_onehot = career_ohe.fit_transform(df[['career_aspiration']].astype(str))

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

X = np.hstack((X_numeric_scaled, career_onehot))

le = LabelEncoder()
y = le.fit_transform(df['stream'])

# Save preprocess objects
with open('app/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('app/career_ohe.pkl', 'wb') as f:
    pickle.dump(career_ohe, f)

with open('app/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Compute class weights for imbalanced classes
weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
weights = torch.tensor(weights, dtype=torch.float32)

# ------------- Define model -------------

class StreamANN(nn.Module):
    def __init__(self, input_dim):
        super(StreamANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(0.3)
        self.output = nn.Linear(16, 4)  # 4 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        return self.output(x)

input_dim = X_train.shape[1]
model = StreamANN(input_dim)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------- Train -------------

best_acc = 0
num_epochs = 200

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        predicted_classes = torch.argmax(preds, dim=1)
        acc = accuracy_score(y_test.numpy(), predicted_classes.numpy())

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'app/model.pth')  # Save best model

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{num_epochs}] Loss: {loss.item():.4f} | Val Accuracy: {acc:.4f}")

print(f"\n✅ Training complete. Best Validation Accuracy: {best_acc:.4f}")
print("Model saved to app/model.pth")
print("Scaler saved to app/scaler.pkl")
print("OneHotEncoder saved to app/career_ohe.pkl")
print("LabelEncoder saved to app/label_encoder.pkl")
# Save background data for SHAP
background = X_train[:100].numpy()
np.save('app/background.npy', background)
print("SHAP background saved to app/background.npy")

