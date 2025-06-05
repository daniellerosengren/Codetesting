import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
n_samples = 10000       # Number of synthetic samples
n_timesteps = 20        # Length of time vector
t = np.linspace(0, 1, n_timesteps)
tau = 0.1               # Fixed interaction time
noise_level = 0.05      # Standard deviation of Gaussian noise

# --- NEURAL NETWORK FROM SCRATCH ---
class Layer:
    def __init__(self, input_size, output_size):
        # Initialize weights with He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        # Store inputs for backpropagation
        self.inputs = inputs
        # Compute weighted sum plus bias
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def backward(self, gradient, learning_rate):
        # Compute gradients
        weights_gradient = np.dot(self.inputs.T, gradient)
        bias_gradient = np.sum(gradient, axis=0, keepdims=True)
        
        # Compute gradient for the next layer
        input_gradient = np.dot(gradient, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
        
    def backward(self, gradient, learning_rate):
        # ReLU gradient: 1 if input > 0, else 0
        return gradient * (self.inputs > 0)

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
        return gradient
    
    def train(self, X, y, epochs, batch_size, learning_rate, X_val=None, y_val=None):
        n_samples = X.shape[0]
        history = {'loss': [], 'val_loss': []}
        
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
                y_pred = self.forward(X_batch)
                
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
                y_val_pred = self.forward(X_val)
                val_loss = np.mean(np.sum((y_val_pred - y_val)**2, axis=1))
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        return self.forward(X)

# --- DATA GENERATION ---
def probability_signal(t, Omega, tau, delta, phi):
    """Generate a single NV-based sine-like signal."""
    return 0.5 + Omega * tau * np.sin(delta * t + phi)

def generate_mixed_dataset(n_samples, noise_level=0.05):
    X = []  # Input signals (mixtures)
    y = []  # Labels: sorted frequencies of the two components

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

        # Store input and sorted output frequencies
        X.append(noisy)
        y.append(sorted([delta1, delta2]))

    return np.array(X), np.array(y)

# --- MAIN EXECUTION ---
# Generate dataset
print("Generating dataset...")
X, y = generate_mixed_dataset(n_samples)

# Normalize input features
print("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Further split training data to get validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Create neural network
print("Creating neural network...")
model = NeuralNetwork()
model.add(Layer(n_timesteps, 64))
model.add(ReLU())
model.add(Layer(64, 64))
model.add(ReLU())
model.add(Layer(64, 2))  # Output: two frequency values

# Train model
print("Training model...")
history = model.train(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    X_val=X_val,
    y_val=y_val
)

# Evaluate on test set
print("Evaluating model...")
y_pred = model.predict(X_test)
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

# Plot predictions vs true values
plt.subplot(1, 2, 2)
plt.plot(y_test[:50, 0], label='True δ1', marker='o')
plt.plot(y_test[:50, 1], label='True δ2', marker='o')
plt.plot(y_pred[:50, 0], label='Pred δ1', marker='x')
plt.plot(y_pred[:50, 1], label='Pred δ2', marker='x')
plt.title("Predicted vs True Frequencies")
plt.xlabel("Sample Index")
plt.ylabel("Frequency (δ)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Visualize a few example signals and predictions
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    
    # Get original (unscaled) signal
    original_signal = X[np.where(np.all(X_scaled == X_test[i], axis=1))[0][0]]
    
    # Plot the signal
    plt.plot(t, original_signal, 'b-', label='Mixed Signal')
    
    # Plot the ground truth and prediction
    true_freqs = y_test[i]
    pred_freqs = y_pred[i]
    
    # Generate clean signals with the true and predicted frequencies
    s1_true = probability_signal(t, 0.6, tau, true_freqs[0], 0)
    s2_true = probability_signal(t, 0.6, tau, true_freqs[1], 0)
    s1_pred = probability_signal(t, 0.6, tau, pred_freqs[0], 0)
    s2_pred = probability_signal(t, 0.6, tau, pred_freqs[1], 0)
    
    plt.plot(t, s1_true, 'g--', label=f'True δ1={true_freqs[0]:.2f}')
    plt.plot(t, s2_true, 'r--', label=f'True δ2={true_freqs[1]:.2f}')
    plt.plot(t, s1_pred, 'g:', label=f'Pred δ1={pred_freqs[0]:.2f}')
    plt.plot(t, s2_pred, 'r:', label=f'Pred δ2={pred_freqs[1]:.2f}')
    
    plt.title(f"Sample {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
