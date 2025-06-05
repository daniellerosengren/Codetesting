import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- CONFIGURATION ---
n_samples = 20000      # Number of synthetic samples
n_timesteps = 20        # Length of time vector
t = np.linspace(0, 1, n_timesteps)
tau = 0.1               # Fixed interaction time
noise_level = 0.05      # Standard deviation of Gaussian noise

# --- NEURAL NETWORK FROM SCRATCH ---
class Layer:
    def __init__(self, input_size, output_size, weight_decay=0.0001):
        # Initialize weights with He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.weight_decay = weight_decay  # L2 regularization parameter
        
    def forward(self, inputs, training=True):
        # Store inputs for backpropagation
        self.inputs = inputs
        # Compute weighted sum plus bias
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def backward(self, gradient, learning_rate):
        # Compute gradients
        weights_gradient = np.dot(self.inputs.T, gradient)
        bias_gradient = np.sum(gradient, axis=0, keepdims=True)
        
        # Add L2 regularization gradient
        weights_gradient += self.weight_decay * self.weights
        
        # Compute gradient for the next layer
        input_gradient = np.dot(gradient, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.where(inputs > 0, inputs, inputs * self.alpha)
        
    def backward(self, gradient, learning_rate):
        return gradient * np.where(self.inputs > 0, 1, self.alpha)

class BatchNormalization:
    def __init__(self, input_dim, epsilon=1e-8, momentum=0.9):
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        
        if training:
            # Calculate batch statistics
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize
            self.x_norm = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            
            # Scale and shift
            return self.gamma * self.x_norm + self.beta
        else:
            # Use running statistics for inference
            x_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * x_norm + self.beta
    
    def backward(self, gradient, learning_rate):
        # Get batch size
        m = self.inputs.shape[0]
        
        # Compute gradients for gamma and beta
        dgamma = np.sum(gradient * self.x_norm, axis=0, keepdims=True)
        dbeta = np.sum(gradient, axis=0, keepdims=True)
        
        # Compute gradient with respect to input
        dx_norm = gradient * self.gamma
        dvar = np.sum(dx_norm * (self.inputs - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + dvar * np.mean(-2 * (self.inputs - self.batch_mean), axis=0, keepdims=True)
        dx = dx_norm / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.inputs - self.batch_mean) / m + dmean / m
        
        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dx

class Dropout:
    def __init__(self, rate=0.2):
        self.rate = rate
        self.mask = None
        
    def forward(self, inputs, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
            return inputs * self.mask
        else:
            return inputs
        
    def backward(self, gradient, learning_rate):
        return gradient * self.mask

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, inputs, training=True):
        for layer in self.layers:
            inputs = layer.forward(inputs, training)
        return inputs
    
    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
        return gradient
    
    def train(self, X, y, epochs, batch_size, learning_rate_init=0.001, X_val=None, y_val=None, patience=10):
        n_samples = X.shape[0]
        history = {'loss': [], 'val_loss': []}
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Learning rate decay
            learning_rate = learning_rate_init / (1 + 0.1 * epoch)
            
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
                gradient = 2 * (y_pred - y_batch) / y_batch.shape[0]
                
                # Backward pass
                self.backward(gradient, learning_rate)
            
            history['loss'].append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val, training=False)
                val_loss = np.mean(np.sum((y_val_pred - y_val)**2, axis=1))
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {learning_rate:.6f}")
                
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
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, LR: {learning_rate:.6f}")
        
        return history
    
    def predict(self, X):
        return self.forward(X, training=False)

# --- DATA GENERATION ---
def probability_signal(t, Omega, tau, delta, phi):
    """Generate a single NV-based sine-like signal."""
    return 0.5 + Omega * tau * np.sin(delta * t + phi)

def generate_mixed_dataset(n_samples, noise_level=0.05):
    X_all = []  # Time vectors
    y_all = []  # Noisy signals
    
    # Store parameters for reference (not used in training)
    params = []  # Will store [delta1, delta2] for each sample

    for _ in range(n_samples):
        # Random parameters for signal 1
        Omega1 = np.random.uniform(0.4, 0.8)
        delta1 = np.random.uniform(1.0, 5.0)
        phi1 = np.random.uniform(0, 2*np.pi)

        # Random parameters for signal 2
        Omega2 = np.random.uniform(0.4, 0.8)
        delta2 = np.random.uniform(1.0, 5.0)
        phi2 = np.random.uniform(0, 2*np.pi)

        # Generate individual signals
        s1 = probability_signal(t, Omega1, tau, delta1, phi1)
        s2 = probability_signal(t, Omega2, tau, delta2, phi2)

        # Combine signals
        mixed = s1 + s2

        # Add Gaussian noise
        noisy = mixed + np.random.normal(0, noise_level, size=mixed.shape)

        # Store time as X and noisy signal as y
        X_all.append(t)
        y_all.append(noisy)
        
        # Store parameters for reference
        params.append(sorted([delta1, delta2]))

    return np.array(X_all), np.array(y_all), np.array(params)

# --- DATA AUGMENTATION ---
def augment_data_modified(X, y, params, augmentation_factor=2):
    n_samples = X.shape[0]
    X_aug = []
    y_aug = []
    params_aug = []
    
    for i in range(n_samples):
        # Add original sample
        X_aug.append(X[i])
        y_aug.append(y[i])
        params_aug.append(params[i])
        
        # Add augmented samples
        for _ in range(augmentation_factor - 1):
            # Add small random noise to create a slightly different sample
            noise = np.random.normal(0, 0.02, size=y[i].shape)
            X_aug.append(X[i])  # Time stays the same
            y_aug.append(y[i] + noise)  # Add noise to signal
            params_aug.append(params[i])  # Parameters stay the same
    
    return np.array(X_aug), np.array(y_aug), np.array(params_aug)

# --- FFT PREPROCESSING ---
def apply_fft(time_vectors, signals):
    """Apply FFT to time-domain signals and extract frequency features."""
    n_samples = signals.shape[0]
    n_timesteps = signals.shape[1]
    
    # Sample spacing (assuming uniform time spacing)
    dt = (time_vectors[0][-1] - time_vectors[0][0]) / (n_timesteps - 1)
    
    # Compute FFT for each signal
    fft_features = []
    for i in range(n_samples):
        # Get the signal and ensure it's a numpy array
        signal = np.asarray(signals[i], dtype=np.float64)
        
        # Compute FFT
        yf = np.fft.fft(signal)
        
        # Compute frequency bins
        xf = np.fft.fftfreq(n_timesteps, dt)
        
        # Only take the positive frequencies (first half of the spectrum)
        n_positive = n_timesteps // 2
        
        # Compute magnitude
        magnitude = 2.0/n_timesteps * np.abs(yf[:n_positive])
        
        # Compute phase
        phase = np.angle(yf[:n_positive])
        
        # Combine magnitude and phase as features
        features = np.concatenate([magnitude, phase])
        fft_features.append(features)
    
    return np.array(fft_features)

# --- MAIN EXECUTION ---
# Generate dataset
print("Generating dataset...")
X, y, params = generate_mixed_dataset(n_samples)

# Data augmentation
print("Augmenting data...")
X, y, params = augment_data_modified(X, y, params, augmentation_factor=2)
print(f"Dataset size after augmentation: {X.shape[0]} samples")

# Apply FFT preprocessing to the signals (y)
print("Applying FFT preprocessing...")
y_fft = apply_fft(X, y)
print(f"FFT features shape: {y_fft.shape}")

# Normalize FFT features
print("Normalizing features...")
scaler = StandardScaler()
y_fft_scaled = scaler.fit_transform(y_fft)

# Train-test split
print("Splitting data...")
X_train, X_test, y_fft_train, y_fft_test, y_train, y_test, params_train, params_test = train_test_split(
    X, y_fft_scaled, y, params, test_size=0.2, random_state=42)

# Further split training data to get validation set
X_train, X_val, y_fft_train, y_fft_val, y_train, y_val, params_train, params_val = train_test_split(
    X_train, y_fft_train, y_train, params_train, test_size=0.1, random_state=42)

# Create neural network with improved architecture
print("Creating neural network...")
model = NeuralNetwork()

# Get input dimension from FFT features
input_dim = y_fft_train.shape[1]

# Add layers
model.add(Layer(input_dim, 128, weight_decay=0.0001))
model.add(BatchNormalization(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Layer(128, 64, weight_decay=0.0001))
model.add(BatchNormalization(64))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Layer(64, n_timesteps))  # Output: signal values at each timestep

# Train model
print("Training model...")
history = model.train(
    y_fft_train, y_train,
    epochs=100,
    batch_size=128,
    learning_rate_init=0.001,
    X_val=y_fft_val,
    y_val=y_val,
    patience=15
)

# Evaluate on test set
print("Evaluating model...")
y_pred = model.predict(y_fft_test)
test_mse = np.mean(np.sum((y_pred - y_test)**2, axis=1))
test_mae = np.mean(np.sum(np.abs(y_pred - y_test), axis=1))
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()

# Plot predictions vs true signals for a few examples
plt.subplot(1, 2, 2)
sample_idx = 0  # Index of the sample to visualize
plt.plot(X_test[sample_idx], y_test[sample_idx], 'b-', label='True Signal')
plt.plot(X_test[sample_idx], y_pred[sample_idx], 'r--', label='Predicted Signal')
plt.title("Predicted vs True Signal")
plt.xlabel("Time")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Visualize a few example signals and their FFT
plt.figure(figsize=(15, 10))
for i in range(4):
    # Original signal
    plt.subplot(4, 2, 2*i+1)
    # Get the time and signal
    time_vector = X_test[i]
    original_signal = y_test[i]
    predicted_signal = y_pred[i]
    
    # Plot the signals
    plt.plot(time_vector, original_signal, 'b-', label='True Signal')
    plt.plot(time_vector, predicted_signal, 'r--', label='Predicted Signal')
    
    # For reference, show the parameters (frequencies) used to generate this signal
    true_freqs = params_test[i]
    
    plt.title(f"Sample {i+1} (δ1={true_freqs[0]:.2f}, δ2={true_freqs[1]:.2f})")
    plt.xlabel("Time")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid()
    
    # FFT magnitude
    plt.subplot(4, 2, 2*i+2)
    dt = (time_vector[-1] - time_vector[0]) / (len(time_vector) - 1)
    
    # Compute FFT for both true and predicted signals
    true_signal = np.asarray(original_signal, dtype=np.float64)
    pred_signal = np.asarray(predicted_signal, dtype=np.float64)
    
    # Use numpy's FFT implementation directly
    yf_true = np.fft.fft(true_signal)
    yf_pred = np.fft.fft(pred_signal)
    xf = np.fft.fftfreq(len(time_vector), dt)
    n_positive = len(time_vector) // 2
    
    # Compute magnitude
    magnitude_true = 2.0/len(time_vector) * np.abs(yf_true[:n_positive])
    magnitude_pred = 2.0/len(time_vector) * np.abs(yf_pred[:n_positive])
    
    # Plot FFT magnitudes
    plt.plot(xf[0:n_positive], magnitude_true, 'b-', label='True Signal FFT')
    plt.plot(xf[0:n_positive], magnitude_pred, 'r--', label='Predicted Signal FFT')
    
    plt.title(f"Sample {i+1} - Frequency Domain")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# Print summary of FFT-based approach
print("\nSummary of FFT-Based Approach:")
print("1. Data Format:")
print("   - X: Time vectors")
print("   - y: Signal values with Gaussian noise")
print("2. Preprocessing:")
print("   - Applied FFT to time-domain signals")
print("   - Extracted magnitude and phase information")
print("   - Combined as feature vector")
print("3. Neural Network:")
print(f"   - Fully connected architecture with 128 → 64 → {n_timesteps} neurons")
print("   - Batch normalization and dropout for regularization")
print("   - LeakyReLU activation functions")
print("   - L2 weight decay")
print("4. Training:")
print("   - Learning rate scheduling")
print("   - Early stopping")
print("   - Mini-batch gradient descent")
