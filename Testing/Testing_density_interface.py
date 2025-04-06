"""
Nanosat Formation Flying Project

Test script for the interfaces with the optimizer for fast density predictions

Author: 
    Hemanth
"""

import numpy as np
from pyatmos import download_sw_nrlmsise00, read_sw_nrlmsise00
from pyatmos import nrlmsise00
import torch
from torch import nn
from torch.optim import adam as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

# Space weather data.
swfile = download_sw_nrlmsise00() 
swdata = read_sw_nrlmsise00(swfile)

base_date= datetime.datetime(2014, 2, 2, 0, 0, 0)
base_date_for_scaling = base_date 


def nrlmsise00_density(inputs):
    densities = []
    for row in inputs:
        time_hours = row[0]       # time in hours (offset from base_date)
        altitude = row[1]         # altitude in km
        latitude = row[2]         # latitude in degrees
        longitude = row[3]        # longitude in degrees
        
        # Creating time string by adding the offset (in hours) to the base_date.
        current_time = base_date + datetime.timedelta(hours=float(time_hours))
        t_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # NRLMSISE-00 model with the computed time and location.
        nrl_output = nrlmsise00(t_str, (latitude, longitude, altitude), swdata)
        
        # Extracting the density (rho) [kg/m^3] from the model output.
        densities.append(nrl_output.rho)
    return np.array(densities).reshape(-1, 1)

# Generate training data.
num_samples = 5000
np.random.seed(42)

# Sample each input uniformly over a range typical for the mission.
time = np.random.uniform(0, 24, (num_samples, 1))                # hours
altitude = np.random.uniform(160, 500, (num_samples, 1))           # km
latitude = np.random.uniform(-90, 90, (num_samples, 1))            # degrees
longitude = np.random.uniform(-180, 180, (num_samples, 1))         # degrees
solar_flux = np.random.uniform(70, 150, (num_samples, 1))          # sfu (example range)
geomagnetic_index = np.random.uniform(0, 10, (num_samples, 1))     # example range

# Stack the input features into a single array.
X = np.hstack([time, altitude, latitude, longitude, solar_flux, geomagnetic_index])
# Compute density values using the NRLMSISE-00 function.
y = nrlmsise00_density(X)

# Reshape y to ensure it has the correct dimensions.
epsilon = 1e-20
y_log = np.log(y + epsilon)

# Scale the input features.
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Scale the log-transformed density.
scaler_y = StandardScaler()
y_log_scaled = scaler_y.fit_transform(y_log)

# Split the dataset into training, validation, and test sets.
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_log_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert numpy arrays to PyTorch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class DensitySurrogate(nn.Module):
    def __init__(self, input_dim):
        super(DensitySurrogate, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )
    
    def forward(self, x):
        return self.model(x)

input_dim = X_train_tensor.shape[1]
model = DensitySurrogate(input_dim)
print(model)

# Define loss function and optimizer.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training settings.
epochs = 200
batch_size = 16

# Create DataLoader for batch processing.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop.
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs_batch, targets_batch in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients.
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Compute validation loss.
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
    
    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f} - Validation Loss: {val_loss:.4f}")

model.eval()
with torch.no_grad():
    new_datetime_str = "2014-02-01 00:12:00"  # Change the date and time as needed.
    # Example values for the new input features.
    new_altitude = 513.612  # km
    new_latitude = -77.062  # degrees
    new_longitude = 111.118 # degrees
    new_solar_flux = 161    # sfu 
    new_geomagnetic_index = 3.3  # kp

    # Use the same base_date as used in training (February 2, 2014)
    base_date_for_scaling = base_date  # datetime(2014, 2, 2, 0, 0, 0)
    dt = datetime.datetime.strptime(new_datetime_str, '%Y-%m-%d %H:%M:%S')
    time_hours = (dt - base_date_for_scaling).total_seconds() / 3600

    # Create the new input array with numerical values.
    new_input_numeric = np.array([[time_hours, new_altitude, new_latitude, new_longitude, new_solar_flux, new_geomagnetic_index]])
    
    # Scale the new input using the previously fitted scaler.
    new_input_scaled = scaler_X.transform(new_input_numeric)
    new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)
    
    # Predict the log-density using the trained model.
    predicted_density_log_scaled = model(new_input_tensor)
    
    # Inverse transform the scaled log-density.
    predicted_log = scaler_y.inverse_transform(predicted_density_log_scaled.detach().numpy())
    
    # Invert the log transformation to obtain the predicted density.
    predicted_density = np.exp(predicted_log)
    print("Predicted Density:", predicted_density[0][0])

# Evaluate on the test set.
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor).item()
    print(f"Test Loss (MSE): {test_loss:.4f}")
