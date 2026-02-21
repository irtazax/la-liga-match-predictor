import torch
import torch.nn as nn
import torch.optim as optim
import data

# Initializing X and Y
X = torch.tensor(data.X, dtype=torch.float32)
Y = torch.tensor(data.Y, dtype=torch.float32)


# Splitting the data into training, validation and test sets
n = len(X)

train_end = int(0.7 * n)
val_end   = int(0.85 * n)

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Making sizes of layers (input size input x hidden, hidden size hidden x output)
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 2)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NeuralNet(X_train.shape[1])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)


# --------------------------------------------------
# Training setup
# --------------------------------------------------

epochs = 400
batch_size = 128

train_losses = []
train_accs = []
val_losses = []
val_accs = []


# --------------------------------------------------
# Training loop

def match_result(goals):
    home, away = goals
    if home > away:
        return 1   # Home win
    elif home < away:
        return -1  # Away win
    else:
        return 0   # Draw
    
def outcome_accuracy(preds, targets):
    correct = 0
    total = len(targets)

    for p, t in zip(preds, targets):
        pred_result = match_result(p)
        true_result = match_result(t)
        if pred_result == true_result:
            correct += 1

    return correct / total


for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))

    epoch_loss = 0.0
    num_batches = 0

    for i in range(0, X_train.size(0), batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], Y_train[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    train_loss = epoch_loss / num_batches

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, Y_val)

        val_preds = torch.round(torch.clamp(val_outputs, min=0))
        val_acc = outcome_accuracy(
            val_preds.cpu().numpy(),
            Y_val.cpu().numpy()
        )

        train_outputs = model(X_train)
        train_preds = torch.round(torch.clamp(train_outputs, min=0))
        train_acc = outcome_accuracy(
            train_preds.cpu().numpy(),
            Y_train.cpu().numpy()
        )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.3f} "
        f"Val Loss: {val_loss.item():.4f} "
        f"Val Acc: {val_acc:.3f}"
    )

    print(
    f"Epoch [{epoch+1}/{epochs}] "
    f"Train Loss: {loss.item():.4f} "
    f"Train Acc: {train_acc:.3f} "
    f"Val Loss: {val_loss.item():.4f} "
    f"Val Acc: {val_acc:.3f}")

torch.save(model.state_dict(), "laliga_model.pt")
train_losses.append(train_loss)
train_accs.append(train_acc)
val_losses.append(val_loss.item())
val_accs.append(val_acc)