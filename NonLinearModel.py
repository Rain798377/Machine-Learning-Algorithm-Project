import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# Windows fallback for colored text (Just for better terminal output)
try:
    from colorama import init as _colorama_init
    from colorama import Fore, Style
    _colorama_init(autoreset=True)
except Exception:
    class _Dummy:
        def __getattr__(self, name):
            return ""
    Fore = _Dummy()
    Style = _Dummy()

# Variables
train_model_epoches_int = int(input("Number of epoches for training model: "))
train_model_learning_rate = float(input("Learning rate for training model: "))
homogeneous_epoches_int = int(input("Number of epoches for homogeneous model(s): "))
homogeneous_models_int = int(input("Number of homogeneous models: "))
heterogeneous_epoches_int = int(input("Number of epoches for heterogeneous model(s): "))
heterogeneous_models_int = int(input("Number of heterogeneous models: "))
refinement_model_epoch = int(input("Number of epoches for refinement model: "))

# Using Cuda or not. This detects if a NVIDIA GPU is available and uses it if possible.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Example sequence (Use for testing)
# data = [1, 3, 6, 10, 15, 21]


# Generating data based on the given formula below. (Sorry! Too lazy to let user input data points one by one :) )
data = []
for i in range(100):
    x1 = i
    x2 = i * 0.5
    x3 = i + 2
    x4 = i / 3
    y = (0.5 * x1**2) - (2.3 * x2) + 4 * math.sin(0.8 * x3) + 3.5 * math.cos(0.5 * x4) + 7
    data.append(y)

# Lets user know the current data set.
print(f"Current data set: {data}")



# Prepare data array for training
X = np.array(data[:-1]).reshape(-1, 1).astype(np.float32)
y = np.array(data[1:]).reshape(-1, 1).astype(np.float32)

# split train and validation data (last 10 points for validation)
val_size = 10
X_train = torch.from_numpy(X[:-val_size]).to(device)
y_train = torch.from_numpy(y[:-val_size]).to(device)
X_val = torch.from_numpy(X[-val_size:]).to(device)
y_val = torch.from_numpy(y[-val_size:]).to(device)


X_tensor = torch.from_numpy(X).to(device)
y_tensor = torch.from_numpy(y).to(device)
last_X = torch.tensor([[data[-1]]], dtype=torch.float32).to(device)

# Feed function for models
class FeedForwardNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training model function (Has a overfitt tollerance feature but disabled for now)
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=train_model_epoches_int, lr=train_model_learning_rate, name="Model", weight_decay=1e-3): #patience=1000,
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5) # reduce LR over time

    #best_val_loss = float('inf')
    #patience_counter = 0
    # Track the last printed train/val loss so we can color the numbers when they improve/worsen
    prev_print_train = None
    prev_print_val = None
    
    for epoch in range(1, epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        # Print model progress with colored components for fun :)
        if epoch % 100 == 0 or epoch == 1:
            # Model name in green
            name_str = f"{Style.BRIGHT}{Fore.GREEN}{name}{Style.RESET_ALL}"
            # Epoch in reddish-orange
            epoch_str = f"{Fore.LIGHTRED_EX}Epoch: {Style.RESET_ALL}{Fore.LIGHTWHITE_EX}{epoch}/{epochs}{Style.RESET_ALL}"

            # Train loss label in light blue
            train_label = f"{Fore.LIGHTBLUE_EX}Train Loss: {Style.RESET_ALL}"
            # Decide color for train loss number
            train_val = loss.item()
            if prev_print_train is None:
                train_num = f"{train_val:.6f}"
            else:
                if train_val < prev_print_train:
                    train_num = f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{train_val:.6f}{Style.RESET_ALL}"
                else:
                    train_num = f"{Style.BRIGHT}{Fore.RED}{train_val:.6f}{Style.RESET_ALL}"

            # Val loss label in light green
            val_label = f"{Fore.GREEN}Val Loss: {Style.RESET_ALL}"
            # Decide color for val loss number
            val_val = val_loss.item()
            if prev_print_val is None:
                val_num = f"{val_val:.6f}"
            else:
                if val_val > prev_print_val:
                    val_num = f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{val_val:.6f}{Style.RESET_ALL}"
                else:
                    val_num = f"{Style.BRIGHT}{Fore.RED}{val_val:.6f}{Style.RESET_ALL}"

            print(f"{name_str} - {epoch_str}, {train_label} {train_num}, {val_label} {val_num}")

            # update previous printed values :)
            prev_print_train = train_val
            prev_print_val = val_val
        
        # Uncomment below to enable early stopping feature

        # Early stopping
        #if val_loss.item() < best_val_loss:
        #    best_val_loss = val_loss.item()
        #    patience_counter = 0
        #    best_model_state = model.state_dict()
        #else:
        #    patience_counter += 1
        
        #if patience_counter >= patience:
        #    print(f"{name} - Early stopping at epoch {epoch}")
        #    break
    
    #model.load_state_dict(best_model_state)
    return model

# Homogeneous ensemble
num_homogeneous_models = homogeneous_models_int
homogeneous_models = []

for i in range(1, num_homogeneous_models + 1):
    model = FeedForwardNN(hidden_size=15).to(device) # defualt hidden: 15 ... 25
    train_model(model, X_train, y_train, X_val, y_val,
                epochs=homogeneous_epoches_int, lr=train_model_learning_rate, name=f"Homogeneous_Model_{i}") #, patience=1500
    homogeneous_models.append(model)

# Heterogeneous ensemble
num_hetero_models = heterogeneous_models_int
hetero_models = []

for i in range(1, num_hetero_models + 1):
    hidden_size = 10 + i * 5
    model = FeedForwardNN(hidden_size=hidden_size).to(device)
    train_model(model, X_train, y_train, X_val, y_val,
                epochs=heterogeneous_epoches_int, lr=train_model_learning_rate, name=f"Heterogeneous_Model_{i}") #, patience=1500
    hetero_models.append(model)

# Combine predictions
homogeneous_preds = torch.stack([m(X_tensor).detach() for m in homogeneous_models], dim=0).mean(dim=0)
heterogeneous_preds = torch.stack([m(X_tensor).detach() for m in hetero_models], dim=0).mean(dim=0)
stacked_preds = torch.cat([homogeneous_preds, heterogeneous_preds], dim=1)

# Refinement model
refine_model = FeedForwardNN(input_size=2, hidden_size=20).to(device) # defualt hidden value: 20 ... 30
train_model(refine_model, stacked_preds[:-1], y_tensor[:-1],
            stacked_preds[-1:], y_tensor[-1:],
            epochs=refinement_model_epoch , lr=train_model_learning_rate, name="Refinement_Model") #, patience=200

# Predict next number
homogeneous_next = torch.stack([m(last_X).detach() for m in homogeneous_models], dim=0).mean(dim=0)
heterogeneous_next = torch.stack([m(last_X).detach() for m in hetero_models], dim=0).mean(dim=0)
refine_input = torch.cat([homogeneous_next, heterogeneous_next], dim=1)
final_prediction = refine_model(refine_input).item()
next_x = 100
x1 = next_x
x2 = next_x * 0.5
x3 = next_x + 2
x4 = next_x / 3
correct_term = (0.5 * x1**2) - (2.3 * x2) + 4 * math.sin(0.8 * x3) + 3.5 * math.cos(0.5 * x4) + 7
abs_error = abs(float(correct_term - final_prediction))


# Print each homogeneous model's prediction and the homogeneous ensemble mean
for idx, m in enumerate(homogeneous_models, start=1):
    pred_val = m(last_X).detach().item()
    model_label = f"{Fore.GREEN}Homogeneous_Model_{idx}{Style.RESET_ALL}"
    print(f"\n{model_label} predicted term: {pred_val:.8f}")
print(f"{Fore.GREEN}Homogeneous ensemble mean{Style.RESET_ALL} predicted term: {homogeneous_next.item():.8f}")

# Print each heterogeneous model's prediction and the heterogeneous ensemble mean
for idx, m in enumerate(hetero_models, start=1):
    pred_val = m(last_X).detach().item()
    model_label = f"{Fore.GREEN}Heterogeneous_Model_{idx}{Style.RESET_ALL}"
    print(f"\n{model_label} predicted term: {pred_val:.8f}")
print(f"{Fore.GREEN}Heterogeneous ensemble mean{Style.RESET_ALL} predicted term: {heterogeneous_next.item():.8f}")


# Final results yippee!
print(f"\nCorrect term: {correct_term}")
print(f"Final refined prediction: {final_prediction:.8f}") # change :.8f to :.2f for rounding to nearest 2 decimal places
print(f"Absolute Error: {abs_error:.8f}")

# Keeps user from terminal auto closing on Windows
input("\nPress Enter to exit...")

# Total time working on this project: ~5 days