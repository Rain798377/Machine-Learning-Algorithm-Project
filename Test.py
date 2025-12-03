import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# This file is a copy of the improved Main.py for standalone testing.

# Variables (interactive prompts kept for parity with Main.py)
train_model_epoches_int = int(input("Number of epoches for training model: "))
train_model_learning_rate = float(input("Learning rate for training model: "))
homogeneous_epoches_int = int(input("Number of epoches for homogeneous model(s): "))
homogeneous_models_int = int(input("Number of homogeneous models: "))
heterogeneous_epoches_int = int(input("Number of epoches for heterogeneous model(s): "))
heterogeneous_models_int = int(input("Number of heterogeneous models: "))
refinement_model_epoch = int(input("Number of epoches for refinement model: "))

# Data generation size (next_x will be equal to this length)
tmp = input("Number of data points to generate (default 100): ")
if tmp.strip() == "":
	data_points = 100
else:
	data_points = int(tmp)

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Using Cuda or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# dataset


data = []
for i in range(data_points):
	x = i
	y = (x**2) * math.sin(3 * x) + 5 * math.cos(x) - 1
	data.append(y)

print(f"Current data set length: {len(data)} (showing first 10 values): {data[:10]}")

# Prepare array
X = np.array(data[:-1]).reshape(-1, 1).astype(np.float32)
y = np.array(data[1:]).reshape(-1, 1).astype(np.float32)

# Train / Val / Test split (time-ordered, no shuffling)
total = X.shape[0]
train_end = max(1, int(total * 0.8))
val_end = train_end + max(1, int(total * 0.1))
if val_end >= total:
	val_end = total - 1

X_train_np = X[:train_end]
y_train_np = y[:train_end]
X_val_np = X[train_end:val_end]
y_val_np = y[train_end:val_end]
X_test_np = X[val_end:]
y_test_np = y[val_end:]

# Scaling (standardize using train stats)
X_mean, X_std = X_train_np.mean(), X_train_np.std()
y_mean, y_std = y_train_np.mean(), y_train_np.std()
if X_std == 0:
	X_std = 1.0
if y_std == 0:
	y_std = 1.0

X_train = torch.from_numpy(((X_train_np - X_mean) / X_std).astype(np.float32)).to(device)
y_train = torch.from_numpy(((y_train_np - y_mean) / y_std).astype(np.float32)).to(device)
X_val = torch.from_numpy(((X_val_np - X_mean) / X_std).astype(np.float32)).to(device)
y_val = torch.from_numpy(((y_val_np - y_mean) / y_std).astype(np.float32)).to(device)
X_test = torch.from_numpy(((X_test_np - X_mean) / X_std).astype(np.float32)).to(device)
y_test = torch.from_numpy(((y_test_np - y_mean) / y_std).astype(np.float32)).to(device)

# Full tensors (scaled) used for ensemble/refinement
X_tensor = torch.from_numpy(((X - X_mean) / X_std).astype(np.float32)).to(device)
y_tensor = torch.from_numpy(((y - y_mean) / y_std).astype(np.float32)).to(device)

# last input (the very last data point, scaled) used to predict the next term
last_X = torch.tensor([[data[-1]]], dtype=torch.float32).to(device)
last_X = (last_X - X_mean) / X_std

# FeedFunc
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

# Training Model
def train_model(model, X_train, y_train, X_val, y_val,
				epochs=train_model_epoches_int, lr=train_model_learning_rate, name="Model", weight_decay=1e-4): #patience=1000,
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	#best_val_loss = float('inf')
	#patience_counter = 0
    
	for epoch in range(1, epochs + 1):
		# Training step
		model.train()
		optimizer.zero_grad()
		outputs = model(X_train)
		loss = criterion(outputs, y_train)
		loss.backward()
		optimizer.step()
        
		# Validation step
		model.eval()
		with torch.no_grad():
			val_outputs = model(X_val)
			val_loss = criterion(val_outputs, y_val)
        
		# Print progress
		if epoch % 50 == 0 or epoch == 1:
			print(f"{name} - Epoch {epoch}/{epochs}, "
				  f"Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
	return model

# ---------------- Step 1: Homogeneous ensemble ----------------
num_homogeneous_models = homogeneous_models_int
homogeneous_models = []

for i in range(1, num_homogeneous_models + 1):
	# choose a modest hidden size that scales with data size
	hidden_size = max(15, min(64, int(len(data) / 10)))
	model = FeedForwardNN(hidden_size=hidden_size).to(device) # default hidden: 15 ... 64
	train_model(model, X_train, y_train, X_val, y_val,
				epochs=homogeneous_epoches_int, lr=train_model_learning_rate, name=f"Homogeneous_Model_{i}") #, patience=1500
	homogeneous_models.append(model)

# ---------------- Step 2: Heterogeneous ensemble ----------------
num_hetero_models = heterogeneous_models_int
hetero_models = []

for i in range(1, num_hetero_models + 1):
	# hetero models have increasing capacity
	hidden_size = min(128, 10 + i * 5 + int(len(data) / 50))
	model = FeedForwardNN(hidden_size=hidden_size).to(device)
	train_model(model, X_train, y_train, X_val, y_val,
				epochs=heterogeneous_epoches_int, lr=train_model_learning_rate, name=f"Heterogeneous_Model_{i}") #, patience=1500
	hetero_models.append(model)

# ---------------- Step 3: Combine predictions ----------------
homogeneous_preds = torch.stack([m(X_tensor).detach() for m in homogeneous_models], dim=0).mean(dim=0)
heterogeneous_preds = torch.stack([m(X_tensor).detach() for m in hetero_models], dim=0).mean(dim=0)
stacked_preds = torch.cat([homogeneous_preds, heterogeneous_preds], dim=1)

# ---------------- Step 4: Refinement model ----------------
refine_model = FeedForwardNN(input_size=2, hidden_size= max(20, min(80, int(len(data)/10)))).to(device) # dynamic hidden
train_model(refine_model, stacked_preds[:X_train.shape[0]], y_tensor[:X_train.shape[0]],
			stacked_preds[X_train.shape[0]:X_train.shape[0]+X_val.shape[0]], y_tensor[X_train.shape[0]:X_train.shape[0]+X_val.shape[0]],
			epochs=refinement_model_epoch , lr=train_model_learning_rate, name="Refinement_Model") #, patience=200

# ---------------- Step 5: Predict next number ----------------
homogeneous_next = torch.stack([m(last_X).detach() for m in homogeneous_models], dim=0).mean(dim=0)
heterogeneous_next = torch.stack([m(last_X).detach() for m in hetero_models], dim=0).mean(dim=0)
refine_input = torch.cat([homogeneous_next, heterogeneous_next], dim=1)
final_prediction_scaled = refine_model(refine_input).item()

# Inverse transform prediction to original scale
final_prediction = final_prediction_scaled * float(y_std) + float(y_mean)

# Compute the true next term. If data contains x=0..(N-1), next_x is N
next_x = len(data)
correct_term = (next_x**2) * math.sin(3 * next_x) + 5 * math.cos(next_x) - 1
abs_error = abs(float(correct_term - final_prediction))

print(f"\nCorrect term (true next value for x={next_x}): {correct_term}")
print(f"\nFinal refined prediction (unscaled): {final_prediction:.8f}") # :.2f for rounding
print(f"\nAbsolute Error: {abs_error:.8f}")
