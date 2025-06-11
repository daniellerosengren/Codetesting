"""
CNN Implementation for Signal Processing with Multiple Signals (No Standardization, No Batch Normalization)
=============================================================

This code implements a 1D Convolutional Neural Network (CNN) from scratch to process
time-series signals. It generates two signals with a frequency offset, saves the data
to CSV files at different stages of processing, and visualizes the results.
This version does not use standardization for data preprocessing or batch normalization in the neural network.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt
import pandas as pd
import os
import time  # Import time module for timing measurements
np.random.seed(42)    # For reproducibility
# --- CONFIGURATION ---
n_samples = 500      # Number of synthetic samples (directly generating more samples instead of augmentation)
n_timesteps = 20      # Length of time vector
t = np.linspace(0, 1, n_timesteps)  # Time points from 0 to 1
tau = 0.1             # Fixed interaction time parameter
noise_level = 0.05    # Standard deviation of Gaussian noise

# --- NEURAL NETWORK FROM SCRATCH WITH SIMPLE CNN IMPLEMENTATION ---

# Step 1: Define the Conv1D layer for 1D convolution operations
class Conv1D:
    """
    1D Convolutional Layer
    
    This layer applies convolution operation on 1D input data (time series).
    It slides a kernel over the input data and computes dot products.
    """
    def __init__(self, filters, kernel_size, input_dim=None, weight_decay=0.0001):
        """
        Initialize the Conv1D layer
        
        Parameters:
        - filters: Number of filters (kernels)
        - kernel_size: Size of each kernel
        - input_dim: Input dimension (optional)
        - weight_decay: L2 regularization parameter
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim if input_dim is not None else 1
        self.weight_decay = weight_decay
        
        # Initialize kernels using He initialization
        self.kernels = np.random.randn(self.filters, self.kernel_size, 1) * np.sqrt(2.0 / (self.kernel_size * 1))
        self.bias = np.zeros((self.filters, 1))
        
        # Initialize placeholders
        self.X = np.zeros((1, 1))  # Will be overwritten in forward pass
        self.output_shape = (1, self.filters, 1)  # Will be overwritten in forward pass
        
    def initialize(self, input_dim):
        """Reinitialize weights if input dimension changes"""
        self.input_dim = input_dim
        self.kernels = np.random.randn(self.filters, self.kernel_size, 1) * np.sqrt(2.0 / (self.kernel_size * 1))
        self.bias = np.zeros((self.filters, 1))
        
    def forward(self, X, training=True):
        """
        Forward pass: Apply convolution operation
        
        Parameters:
        - X: Input data of shape (batch_size, seq_len)
        - training: Whether in training mode
        
        Returns:
        - Output of shape (batch_size, filters * output_len)
        """
        self.X = X
        batch_size, seq_len = X.shape
        
        # Calculate output dimensions
        output_len = seq_len - self.kernel_size + 1
        output = np.zeros((batch_size, self.filters, output_len))
        
        # Reshape X for convolution: (batch_size, 1, seq_len)
        X_reshaped = X.reshape(batch_size, 1, seq_len)
        
        # Perform convolution
        for i in range(batch_size):
            for j in range(self.filters):
                for k in range(output_len):
                    # Extract the window
                    window = X_reshaped[i, :, k:k+self.kernel_size]
                    # Apply convolution - extract scalar values to avoid deprecation warning
                    output[i, j, k] = np.sum(window * self.kernels[j]) + self.bias[j, 0]
        
        # Flatten the output: (batch_size, filters * output_len)
        self.output_shape = (batch_size, self.filters, output_len)
        return output.reshape(batch_size, -1)
    
    def backward(self, dY, learning_rate):
        """
        Backward pass: Compute gradients and update parameters
        
        Parameters:
        - dY: Gradient from next layer
        - learning_rate: Learning rate for parameter updates
        
        Returns:
        - Gradient with respect to input
        """
        batch_size, seq_len = self.X.shape
        _, filters, output_len = self.output_shape
        
        # Reshape dY to match output shape: (batch_size, filters, output_len)
        dY = dY.reshape(self.output_shape)
        
        # Initialize gradients
        dX = np.zeros_like(self.X)
        dKernels = np.zeros_like(self.kernels)
        dBias = np.zeros_like(self.bias)
        
        # Reshape X for convolution: (batch_size, 1, seq_len)
        X_reshaped = self.X.reshape(batch_size, 1, seq_len)
        
        # Compute gradients
        for i in range(batch_size):
            for j in range(filters):
                for k in range(output_len):
                    # Gradient for kernels
                    window = X_reshaped[i, :, k:k+self.kernel_size]
                    # Ensure proper broadcasting by reshaping window to match kernel shape
                    window_reshaped = window.reshape(self.kernels[j].shape)
                    dKernels[j] += dY[i, j, k] * window_reshaped
                    
                    # Gradient for bias
                    dBias[j, 0] += dY[i, j, k]
                    
                    # Gradient for input
                    kernel_flat = self.kernels[j].reshape(-1)
                    dX[i, k:k+self.kernel_size] += dY[i, j, k] * kernel_flat
        
        # Add L2 regularization gradient for kernels
        dKernels += self.weight_decay * self.kernels
        
        # Update parameters
        self.kernels -= learning_rate * dKernels
        self.bias -= learning_rate * dBias
        
        return dX

# Step 2: Define the MaxPooling1D layer for downsampling
class MaxPooling1D:
    """
    Max Pooling Layer for 1D data
    
    This layer performs downsampling by taking the maximum value in each window.
    It reduces the spatial dimensions and helps in extracting dominant features.
    """
    def __init__(self, pool_size, stride=None):
        """
        Initialize the MaxPooling1D layer
        
        Parameters:
        - pool_size: Size of the pooling window
        - stride: Stride of the pooling operation (default: same as pool_size)
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        
    def forward(self, X, training=True):
        """
        Forward pass: Apply max pooling operation
        
        Parameters:
        - X: Input data of shape (batch_size, seq_len)
        - training: Whether in training mode
        
        Returns:
        - Output of shape (batch_size, output_len)
        """
        self.X = X
        batch_size, seq_len = X.shape
        
        # Calculate output dimensions
        output_len = (seq_len - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, output_len))
        
        # Store indices for backpropagation
        self.max_indices = np.zeros((batch_size, output_len), dtype=int)
        
        # Perform max pooling
        for i in range(batch_size):
            for j in range(output_len):
                start = j * self.stride
                end = start + self.pool_size
                segment = X[i, start:end]
                max_idx = np.argmax(segment)
                self.max_indices[i, j] = start + max_idx
                output[i, j] = segment[max_idx]
        
        return output
    
    def backward(self, dY, learning_rate):
        """
        Backward pass: Compute gradients
        
        Parameters:
        - dY: Gradient from next layer
        - learning_rate: Learning rate (not used in this layer)
        
        Returns:
        - Gradient with respect to input
        """
        batch_size, seq_len = self.X.shape
        _, output_len = dY.shape
        
        # Initialize gradient for input
        dX = np.zeros_like(self.X)
        
        # Distribute gradient to max indices
        for i in range(batch_size):
            for j in range(output_len):
                dX[i, self.max_indices[i, j]] += dY[i, j]
        
        return dX

# Step 3: Define the Dense (Fully Connected) layer
class Dense:
    """
    Fully Connected (Dense) Layer
    
    This layer connects every input neuron to every output neuron.
    It's typically used in the final layers of a CNN for classification or regression.
    """
    def __init__(self, units, input_dim=None, weight_decay=0.0001):
        """
        Initialize the Dense layer
        
        Parameters:
        - units: Number of output neurons
        - input_dim: Number of input features (optional)
        - weight_decay: L2 regularization parameter
        """
        self.units = units
        self.input_dim = input_dim if input_dim is not None else 1
        self.weight_decay = weight_decay
        
        # Initialize weights using He initialization
        self.weights = np.random.randn(self.input_dim, self.units) * np.sqrt(2.0 / self.input_dim)
        self.bias = np.zeros((1, self.units))
        self.X = np.zeros((1, self.input_dim))  # Will be overwritten in forward pass
        
    def initialize(self, input_dim):
        """Reinitialize weights if input dimension changes"""
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, self.units))
        
    def forward(self, X, training=True):
        """
        Forward pass: Apply linear transformation
        
        Parameters:
        - X: Input data of shape (batch_size, input_dim)
        - training: Whether in training mode
        
        Returns:
        - Output of shape (batch_size, units)
        """
        # Update input_dim if necessary and reinitialize weights
        if X.shape[1] != self.input_dim:
            self.initialize(X.shape[1])
            
        self.X = X
        
        return np.dot(X, self.weights) + self.bias
    
    def backward(self, dY, learning_rate):
        """
        Backward pass: Compute gradients and update parameters
        
        Parameters:
        - dY: Gradient from next layer
        - learning_rate: Learning rate for parameter updates
        
        Returns:
        - Gradient with respect to input
        """
        # Compute gradients
        dW = np.dot(self.X.T, dY)
        dB = np.sum(dY, axis=0, keepdims=True)
        dX = np.dot(dY, self.weights.T)
        
        # Add L2 regularization gradient
        dW += self.weight_decay * self.weights
        
        # Update parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * dB
        
        return dX

# Step 4: Define activation functions

# ReLU Activation
class ReLU:
    """
    Rectified Linear Unit (ReLU) Activation
    
    This activation function returns x for x > 0 and 0 for x <= 0.
    It helps introduce non-linearity and is computationally efficient.
    """
    def forward(self, X, training=True):
        """Forward pass: Apply ReLU activation"""
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dY, learning_rate):
        """Backward pass: Compute gradients"""
        return dY * (self.X > 0)

# LeakyReLU Activation
class LeakyReLU:
    """
    Leaky ReLU Activation
    
    Similar to ReLU but allows a small gradient when the unit is not active.
    It helps prevent "dying ReLU" problem where neurons can get stuck during training.
    """
    def __init__(self, alpha=0.01):
        """
        Initialize LeakyReLU
        
        Parameters:
        - alpha: Slope for negative inputs
        """
        self.alpha = alpha
        
    def forward(self, X, training=True):
        """Forward pass: Apply LeakyReLU activation"""
        self.X = X
        return np.where(X > 0, X, X * self.alpha)
    
    def backward(self, dY, learning_rate):
        """Backward pass: Compute gradients"""
        return dY * np.where(self.X > 0, 1, self.alpha)

# Step 5: Define regularization techniques

# Dropout Layer
class Dropout:
    """
    Dropout Layer
    
    This layer randomly sets a fraction of inputs to zero during training.
    It helps prevent overfitting by making the network more robust.
    """
    def __init__(self, rate=0.2):
        """
        Initialize Dropout
        
        Parameters:
        - rate: Fraction of inputs to drop (between 0 and 1)
        """
        self.rate = rate
        self.mask = None
        
    def forward(self, X, training=True):
        """
        Forward pass: Apply dropout
        
        Parameters:
        - X: Input data
        - training: Whether in training mode
        
        Returns:
        - Output with dropout applied (if in training mode)
        """
        if training:
            # Generate binary mask with probability (1 - rate)
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape) / (1 - self.rate)
            return X * self.mask
        else:
            return X
    
    def backward(self, dY, learning_rate):
        """Backward pass: Compute gradients"""
        return dY * self.mask

# Step 6: Define the neural network model
class SimpleNN:
    """
    Simple Neural Network Model
    
    This class implements a neural network model that can be built by adding layers.
    It handles forward and backward passes, as well as training and prediction.
    """
    def __init__(self):
        """Initialize an empty neural network"""
        self.layers = []
        
    def add(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)
        
    def forward(self, X, training=True):
        """
        Forward pass through all layers
        
        Parameters:
        - X: Input data
        - training: Whether in training mode
        
        Returns:
        - Output of the network
        """
        for layer in self.layers:
            X = layer.forward(X, training)
        return X
    
    def backward(self, dY, learning_rate):
        """
        Backward pass through all layers
        
        Parameters:
        - dY: Gradient of loss with respect to output
        - learning_rate: Learning rate for parameter updates
        
        Returns:
        - Gradient with respect to input
        """
        for layer in reversed(self.layers):
            dY = layer.backward(dY, learning_rate)
        return dY
    
    def train(self, X, y, epochs, batch_size, learning_rate=0.001, X_val=None, y_val=None, patience=10):
        """
        Train the neural network
        
        Parameters:
        - X: Training data
        - y: Training targets
        - epochs: Number of training epochs
        - batch_size: Size of mini-batches
        - learning_rate: Learning rate for parameter updates
        - X_val: Validation data (optional)
        - y_val: Validation targets (optional)
        - patience: Number of epochs to wait for improvement before early stopping
        
        Returns:
        - Training history
        """
        n_samples = X.shape[0]
        history = {'loss': [], 'val_loss': []}
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch, training=True)
                
                # Compute loss (Mean Squared Error)
                batch_loss = np.mean(np.sum((y_pred - y_batch)**2, axis=1))
                epoch_loss += batch_loss * len(X_batch) / n_samples
                
                # Compute gradient of loss with respect to predictions
                dY = 2 * (y_pred - y_batch) / y_batch.shape[0]
                
                # Backward pass
                self.backward(dY, learning_rate)
            
            history['loss'].append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val, training=False)
                val_loss = np.mean(np.sum((y_val_pred - y_val)**2, axis=1))
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.forward(X, training=False)

# Step 7: Define data generation functions

def probability_signal(t, Omega, tau, delta, phi):
    """
    Generate a single NV-based sine-like signal
    
    Parameters:
    - t: Time points
    - Omega: Amplitude parameter
    - tau: Interaction time
    - delta: Frequency parameter
    - phi: Phase parameter
    
    Returns:
    - Signal values
    """
    return 0.5 + Omega * tau * np.sin(delta * t + phi)

def generate_dataset(n_samples, noise_level=0.05, save_to_csv=True):
    """
    Generate a synthetic dataset of signals with two components:
    1. First signal with random parameters
    2. Second signal with frequency offset by 0.05 from the first
    
    Parameters:
    - n_samples: Number of samples to generate
    - noise_level: Standard deviation of Gaussian noise
    - save_to_csv: Whether to save the generated data to a CSV file
    
    Returns:
    - X: Time values (input)
    - y: Signal values with noise (output)
    """
    X_all = []  # Time values (input)
    y_all = []  # Probability signal + noise (output)
    
    # For CSV export
    all_data = []
    
    for i in range(n_samples):
        # Random parameters for first signal
        Omega1 = np.random.uniform(0.4, 0.8)
        delta1 = np.random.uniform(1.0, 5.0)
        phi1 = np.random.uniform(0, 2*np.pi)
        
        # Parameters for second signal (offset in frequency)
        Omega2 = Omega1  # Same amplitude
        delta2 = delta1 + 0.5  # Offset frequency by 0.5
        phi2 = phi1  # Same phase
        
        # Generate clean signals
        signal1 = probability_signal(t, Omega1, tau, delta1, phi1)
        signal2 = probability_signal(t, Omega2, tau, delta2, phi2)
        
        # Mix the signals (simple addition)
        mixed_signal = (signal1 + signal2)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, size=mixed_signal.shape)
        noisy_signal = mixed_signal + noise
        
        # Store time as input and noisy signal as output
        X_all.append(t)
        y_all.append(noisy_signal)
        
        # Store data for CSV export
        if save_to_csv:
            for j in range(len(t)):
                all_data.append({
                    'sample_id': i,
                    'time_point': j,
                    'time_value': t[j],
                    'signal1': signal1[j],
                    'signal2': signal2[j],
                    'mixed_signal': mixed_signal[j],
                    'noise': noise[j],
                    'noisy_signal': noisy_signal[j]
                })
    
    # Save to CSV if requested
    if save_to_csv:
        df = pd.DataFrame(all_data)
        os.makedirs('output', exist_ok=True)
        csv_path = 'output/generated_signals.csv'
        df.to_csv(csv_path, index=False)
        print(f"Generated signals saved to {csv_path}")

    return np.array(X_all), np.array(y_all)

# --- MAIN EXECUTION ---
print("\n" + "="*50)
print("CNN IMPLEMENTATION WITHOUT STANDARDIZATION AND BATCH NORMALIZATION")
print("="*50 + "\n")

# Initialize timing dictionary
timing = {}

# Step 8: Generate and prepare the dataset
print("Generating dataset...")
start_time = time.time()
# Generate samples directly without augmentation
X, y = generate_dataset(n_samples)
timing['data_generation'] = time.time() - start_time
print(f"Dataset size: {X.shape[0]} samples")
print(f"Data generation time: {timing['data_generation']:.2f} seconds")

# Step 9: Skip normalization - use raw data
print("Skipping standardization - using raw data...")
start_time = time.time()
# Use raw data without standardization
X_scaled = X.copy()
y_scaled = y.copy()
timing['data_preparation'] = time.time() - start_time
print(f"Data preparation time: {timing['data_preparation']:.2f} seconds")

# Save raw data to CSV
raw_data = []
for i in range(X_scaled.shape[0]):
    for j in range(X_scaled.shape[1]):
        raw_data.append({
            'sample_id': i,
            'time_point': j,
            'X_raw': X_scaled[i, j],
            'y_raw': y_scaled[i, j]
        })

df_raw = pd.DataFrame(raw_data)
os.makedirs('output', exist_ok=True)
raw_path = 'output/raw_data_no_std_no_batch.csv'
df_raw.to_csv(raw_path, index=False)
print(f"Raw data saved to {raw_path}")

# Step 10: Split the data into training, validation, and test sets
print("Splitting data...")
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
timing['data_splitting'] = time.time() - start_time
print(f"Data splitting time: {timing['data_splitting']:.2f} seconds")

# Step 11: Create and configure the CNN model
print("Creating CNN neural network without batch normalization...")
start_time = time.time()
model = SimpleNN()

# First convolutional block
model.add(Conv1D(filters=16, kernel_size=3))  # 16 filters, kernel size 3
model.add(LeakyReLU(alpha=0.1))               # LeakyReLU activation
# No batch normalization here

# Second convolutional block
model.add(Conv1D(filters=32, kernel_size=3))  # 32 filters, kernel size 3
model.add(LeakyReLU(alpha=0.1))               # LeakyReLU activation
model.add(MaxPooling1D(pool_size=2))          # Max pooling with pool size 2
model.add(Dropout(0.2))                       # Dropout with rate 0.2

# First fully connected layer
model.add(Dense(units=64))                    # 64 hidden units
model.add(LeakyReLU(alpha=0.1))               # LeakyReLU activation
# No batch normalization here
model.add(Dropout(0.2))                       # Dropout with rate 0.2

# Output layer
model.add(Dense(units=n_timesteps))           # Output size matches the number of time steps
timing['model_creation'] = time.time() - start_time
print(f"Model creation time: {timing['model_creation']:.2f} seconds")

# Step 12: Train the model
print("Training model...")
start_time = time.time()
history = model.train(
    X_train, y_train,
    epochs=100,                                # Number of training epochs
    batch_size=128,                           # Batch size
    learning_rate=0.001,                      # Learning rate
    X_val=X_val,                              # Validation data
    y_val=y_val,                              # Validation targets
    patience=15                               # Early stopping patience
)
timing['model_training'] = time.time() - start_time
print(f"Model training time: {timing['model_training']:.2f} seconds")

# Step 13: Evaluate the model and save CNN processed data
print("Evaluating model...")
start_time = time.time()
y_pred = model.predict(X_test)

# Calculate metrics (using raw values)
test_mse = np.mean(np.sum((y_pred - y_test)**2, axis=1))
test_mae = np.mean(np.sum(np.abs(y_pred - y_test), axis=1))

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test.reshape(-1), y_pred.reshape(-1))

# Calculate explained variance score
exp_var = explained_variance_score(y_test.reshape(-1), y_pred.reshape(-1))

# Calculate Mean Absolute Percentage Error (MAPE)
# Add a small epsilon to avoid division by zero
epsilon = 1e-10
mape = np.mean(np.abs((y_test.reshape(-1) - y_pred.reshape(-1)) / (np.abs(y_test.reshape(-1)) + epsilon))) * 100

# Calculate accuracy-like metric (percentage of predictions within a certain threshold)
threshold = 0.1  # 10% of the true value
within_threshold = np.mean(np.abs((y_test.reshape(-1) - y_pred.reshape(-1)) / (np.abs(y_test.reshape(-1)) + epsilon)) < threshold) * 100

print(f"Test MSE (raw): {test_mse:.4f}")
print(f"Test MAE (raw): {test_mae:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Explained variance: {exp_var:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"Predictions within {threshold*100}% threshold: {within_threshold:.2f}%")
timing['model_evaluation'] = time.time() - start_time
print(f"Model evaluation time: {timing['model_evaluation']:.2f} seconds")

# Save CNN processed data to CSV
cnn_processed_data = []
for i in range(X_test.shape[0]):
    for j in range(n_timesteps):
        cnn_processed_data.append({
            'sample_id': i,
            'time_point': j,
            'X_test': X_test[i, j],
            'y_test': y_test[i, j],
            'y_pred': y_pred[i, j]
        })

df_cnn_processed = pd.DataFrame(cnn_processed_data)
os.makedirs('output', exist_ok=True)
cnn_processed_path = 'output/cnn_processed_data_no_std_no_batch.csv'
df_cnn_processed.to_csv(cnn_processed_path, index=False)
print(f"CNN processed data saved to {cnn_processed_path}")

# Step 14: Visualize the results
plt.figure(figsize=(12, 8))

# Plot training history
plt.subplot(2, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()

# Plot predictions vs true values for a few examples
for i in range(3):
    plt.subplot(2, 2, i+2)
    
    # Use original time values
    original_time = t
    
    # Plot the true signal and prediction
    plt.plot(original_time, y_test[i], 'b-', label='True Signal')
    plt.plot(original_time, y_pred[i], 'r--', label='Predicted Signal')
    
    plt.title(f"Sample {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Probability Signal")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig('output/cnn_results_no_std_no_batch.png')
print("Results visualization saved to output/cnn_results_no_std_no_batch.png")
plt.show()

# Step 15: Print model summary
print("\nSummary of CNN Implementation (No Standardization, No Batch Normalization):")
print("1. Data Generation:")
print("   - Generated two signals with 0.5 frequency offset")
print("   - Mixed signals and added noise")
print("   - Saved raw signals to CSV")
print("2. Data Preprocessing:")
print("   - Used raw data without standardization")
print("   - Saved raw data to CSV")
print("3. CNN Architecture:")
print("   - First Conv1D: 16 filters, kernel size 3")
print("   - Second Conv1D: 32 filters, kernel size 3")
print("   - MaxPooling1D with pool size 2")
print("   - Dense layer with 64 units")
print("   - Output layer with 20 units")
print("4. Regularization:")
print("   - No Batch Normalization")
print("   - Dropout (20%)")
print("   - L2 weight decay")
print("5. Results:")
print(f"   - Test MSE: {test_mse:.4f}")
print(f"   - Test MAE: {test_mae:.4f}")
print(f"   - R-squared: {r2:.4f}")
print(f"   - Explained variance: {exp_var:.4f}")
print(f"   - MAPE: {mape:.2f}%")
print(f"   - Predictions within {threshold*100}% threshold: {within_threshold:.2f}%")
print("   - Saved CNN processed data to CSV")
print("\nAll data files are saved in the 'output' directory.")

print("\nTiming Summary:")
print(f"Data Generation:    {timing['data_generation']:.2f} seconds")
print(f"Data Preparation:   {timing['data_preparation']:.2f} seconds")
print(f"Data Splitting:     {timing['data_splitting']:.2f} seconds")
print(f"Model Creation:     {timing['model_creation']:.2f} seconds")
print(f"Model Training:     {timing['model_training']:.2f} seconds")
print(f"Model Evaluation:   {timing['model_evaluation']:.2f} seconds")
print(f"Total Time:         {sum(timing.values()):.2f} seconds")
