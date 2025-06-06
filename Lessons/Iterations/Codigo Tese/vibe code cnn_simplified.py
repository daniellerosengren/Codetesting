import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
n_samples = 2500      # Number of synthetic samples (will be 5000 after augmentation)
n_timesteps = 20        # Length of time vector
t = np.linspace(0, 1, n_timesteps)
tau = 0.1               # Fixed interaction time
noise_level = 0.05      # Standard deviation of Gaussian noise

# --- NEURAL NETWORK FROM SCRATCH WITH SIMPLE CNN IMPLEMENTATION ---
class Conv1D:
    def __init__(self, filters, kernel_size, input_dim=None, weight_decay=0.0001):
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim if input_dim is not None else 1
        self.weight_decay = weight_decay
        
        # Initialize with default values to avoid None
        self.kernels = np.random.randn(self.filters, self.kernel_size, 1) * np.sqrt(2.0 / (self.kernel_size * 1))
        self.bias = np.zeros((self.filters, 1))
        # Initialize with empty arrays to avoid None errors
        self.X = np.zeros((1, 1))  # Will be overwritten in forward pass
        self.output_shape = (1, self.filters, 1)  # Will be overwritten in forward pass
        
    def initialize(self, input_dim):
        self.input_dim = input_dim
        # Initialize kernels with shape (filters, kernel_size, input_dim)
        self.kernels = np.random.randn(self.filters, self.kernel_size, 1) * np.sqrt(2.0 / (self.kernel_size * 1))
        self.bias = np.zeros((self.filters, 1))
        
    def forward(self, X, training=True):
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

class MaxPooling1D:
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        
    def forward(self, X, training=True):
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
        batch_size, seq_len = self.X.shape
        _, output_len = dY.shape
        
        # Initialize gradient for input
        dX = np.zeros_like(self.X)
        
        # Distribute gradient to max indices
        for i in range(batch_size):
            for j in range(output_len):
                dX[i, self.max_indices[i, j]] += dY[i, j]
        
        return dX

class Dense:
    def __init__(self, units, input_dim=None, weight_decay=0.0001):
        self.units = units
        self.input_dim = input_dim if input_dim is not None else 1  # Default to 1 if None
        self.weight_decay = weight_decay
        
        # Initialize with default values to avoid None errors
        self.weights = np.random.randn(self.input_dim, self.units) * np.sqrt(2.0 / self.input_dim)
        self.bias = np.zeros((1, self.units))
        self.X = np.zeros((1, self.input_dim))  # Will be overwritten in forward pass
        
    def initialize(self, input_dim):
        self.input_dim = input_dim
        # He initialization
        self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, self.units))
        
    def forward(self, X, training=True):
        # Update input_dim if necessary and reinitialize weights
        if X.shape[1] != self.input_dim:
            self.initialize(X.shape[1])
            
        self.X = X
        
        return np.dot(X, self.weights) + self.bias
    
    def backward(self, dY, learning_rate):
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

class ReLU:
    def forward(self, X, training=True):
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dY, learning_rate):
        return dY * (self.X > 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, X, training=True):
        self.X = X
        return np.where(X > 0, X, X * self.alpha)
    
    def backward(self, dY, learning_rate):
        return dY * np.where(self.X > 0, 1, self.alpha)

class Dropout:
    def __init__(self, rate=0.2):
        self.rate = rate
        self.mask = None
        
    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape) / (1 - self.rate)
            return X * self.mask
        else:
            return X
    
    def backward(self, dY, learning_rate):
        return dY * self.mask

class BatchNormalization:
    def __init__(self, input_dim=None, epsilon=1e-8, momentum=0.9):
        self.input_dim = input_dim if input_dim is not None else 1  # Default to 1 if None
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize with default values to avoid None errors
        self.gamma = np.ones((1, self.input_dim))
        self.beta = np.zeros((1, self.input_dim))
        self.running_mean = np.zeros((1, self.input_dim))
        self.running_var = np.ones((1, self.input_dim))
        
        # Initialize with empty arrays to avoid None errors
        self.X = np.zeros((1, self.input_dim))  # Will be overwritten in forward pass
        self.X_norm = np.zeros((1, self.input_dim))  # Will be overwritten in forward pass
        self.batch_mean = np.zeros((1, self.input_dim))  # Will be overwritten in forward pass
        self.batch_var = np.ones((1, self.input_dim))  # Will be overwritten in forward pass
        
    def initialize(self, input_dim):
        self.input_dim = input_dim
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        
    def forward(self, X, training=True):
        # Update input_dim if necessary and reinitialize parameters
        if X.shape[1] != self.input_dim:
            self.initialize(X.shape[1])
        
        self.X = X
        
        if training:
            # Calculate batch statistics
            self.batch_mean = np.mean(X, axis=0, keepdims=True)
            self.batch_var = np.var(X, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize
            self.X_norm = (X - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            
            # Scale and shift
            return self.gamma * self.X_norm + self.beta
        else:
            # Use running statistics for inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * X_norm + self.beta
    
    def backward(self, dY, learning_rate):
        # Get batch size
        m = self.X.shape[0]
        
        # Compute gradients for gamma and beta
        dgamma = np.sum(dY * self.X_norm, axis=0, keepdims=True)
        dbeta = np.sum(dY, axis=0, keepdims=True)
        
        # Compute gradient with respect to input
        dX_norm = dY * self.gamma
        dvar = np.sum(dX_norm * (self.X - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
        dmean = np.sum(dX_norm * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + dvar * np.mean(-2 * (self.X - self.batch_mean), axis=0, keepdims=True)
        dX = dX_norm / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.X - self.batch_mean) / m + dmean / m
        
        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dX

class SimpleNN:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training)
        return X
    
    def backward(self, dY, learning_rate):
        for layer in reversed(self.layers):
            dY = layer.backward(dY, learning_rate)
        return dY
    
    def train(self, X, y, epochs, batch_size, learning_rate=0.001, X_val=None, y_val=None, patience=10):
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
                
                # Compute loss
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
        return self.forward(X, training=False)

# --- DATA GENERATION ---
def probability_signal(t, Omega, tau, delta, phi):
    """Generate a single NV-based sine-like signal."""
    return 0.5 + Omega * tau * np.sin(delta * t + phi)

def generate_dataset(n_samples, noise_level=0.05):
    X_all = []  # Time values (input)
    y_all = []  # Probability signal + noise (output)

    for _ in range(n_samples):
        # Random parameters
        Omega = np.random.uniform(0.4, 0.8)
        delta = np.random.uniform(1.0, 5.0)
        phi = np.random.uniform(0, 2*np.pi)

        # Generate clean signal
        clean_signal = probability_signal(t, Omega, tau, delta, phi)
        
        # Add Gaussian noise
        noisy_signal = clean_signal + np.random.normal(0, noise_level, size=clean_signal.shape)
        
        # Store time as input and noisy signal as output
        X_all.append(t)
        y_all.append(noisy_signal)

    return np.array(X_all), np.array(y_all)

# --- DATA AUGMENTATION ---
def augment_data(X, y, augmentation_factor=2):
    n_samples = X.shape[0]
    X_aug = []
    y_aug = []
    
    for i in range(n_samples):
        # Add original sample
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Add augmented samples
        for _ in range(augmentation_factor - 1):
            # Add small random noise to the output signal
            noise = np.random.normal(0, 0.02, size=y[i].shape)
            X_aug.append(X[i])  # Time stays the same
            y_aug.append(y[i] + noise)  # Add noise to the signal
    
    return np.array(X_aug), np.array(y_aug)

# --- MAIN EXECUTION ---
# Generate dataset
print("Generating dataset...")
X, y = generate_dataset(n_samples)

# Data augmentation
print("Augmenting data...")
X, y = augment_data(X, y, augmentation_factor=2)
print(f"Dataset size after augmentation: {X.shape[0]} samples")

# Normalize input features (time values)
print("Normalizing features...")
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize output values (signal values)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Further split training data to get validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Create neural network with CNN architecture
print("Creating CNN neural network...")
model = SimpleNN()

# Add convolutional layers
model.add(Conv1D(filters=16, kernel_size=3))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Conv1D(filters=32, kernel_size=3))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Add fully connected layers
model.add(Dense(units=64))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(units=n_timesteps))  # Output: probability signal values for each time point

# Train model
print("Training model...")
history = model.train(
    X_train, y_train,
    epochs=50,  # Reduced number of epochs
    batch_size=128,
    learning_rate=0.001,
    X_val=X_val,
    y_val=y_val,
    patience=15
)

# Evaluate on test set
print("Evaluating model...")
y_pred = model.predict(X_test)

# Inverse transform to get original scale
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

test_mse = np.mean(np.sum((y_pred - y_test)**2, axis=1))
test_mae = np.mean(np.sum(np.abs(y_pred - y_test), axis=1))
print(f"Test MSE (scaled): {test_mse:.4f}")
print(f"Test MAE (scaled): {test_mae:.4f}")

# Plot training history
plt.figure(figsize=(12, 8))
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
    
    # Get original time values
    original_time = scaler_X.inverse_transform(X_test[i].reshape(1, -1)).flatten()
    
    # Plot the true signal and prediction
    plt.plot(original_time, y_test_original[i], 'b-', label='True Signal')
    plt.plot(original_time, y_pred_original[i], 'r--', label='Predicted Signal')
    
    plt.title(f"Sample {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Probability Signal")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# Print summary of CNN implementation
print("\nSummary of CNN Implementation:")
print("1. Data Processing:")
print("   - Data normalization using StandardScaler")
print("2. Convolutional Layers:")
print("   - First Conv1D: 16 filters, kernel size 3")
print("   - Second Conv1D: 32 filters, kernel size 3")
print("3. Pooling Layer:")
print("   - MaxPooling1D with pool size 2")
print("4. Fully Connected Layers:")
print(f"   - 64 units → {n_timesteps} output neurons")
print("5. Regularization:")
print("   - Batch Normalization after first conv and dense layers")
print("   - Dropout (20%) after pooling and first fully connected layer")
print("   - L2 weight decay (0.0001)")
print("6. Activation Functions:")
print("   - LeakyReLU (alpha=0.1) throughout the network")
print("\nThis model takes normalized time values as input and predicts the probability signal plus noise (y).")
