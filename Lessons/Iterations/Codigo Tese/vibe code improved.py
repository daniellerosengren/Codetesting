import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
n_samples = 20000      # Increased number of samples
n_timesteps = 20        # Length of time vector
t = np.linspace(0, 1, n_timesteps)
tau = 0.1               # Fixed interaction time
noise_level = 0.05      # Standard deviation of Gaussian noise

# --- NEURAL NETWORK FROM SCRATCH WITH IMPROVEMENTS ---
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
    
    def backward(self, gradient, learning_rate, optimizer_state=None):
        # Compute gradients
        weights_gradient = np.dot(self.inputs.T, gradient)
        bias_gradient = np.sum(gradient, axis=0, keepdims=True)
        
        # Add L2 regularization gradient
        weights_gradient += self.weight_decay * self.weights
        
        # Compute gradient for the next layer
        input_gradient = np.dot(gradient, self.weights.T)
        
        # Update parameters using Adam optimizer
        if optimizer_state is not None:
            # Unpack optimizer state
            m_w, v_w, m_b, v_b, t = optimizer_state
            
            # Update biased first moment estimate
            m_w = 0.9 * m_w + 0.1 * weights_gradient
            m_b = 0.9 * m_b + 0.1 * bias_gradient
            
            # Update biased second raw moment estimate
            v_w = 0.999 * v_w + 0.001 * np.square(weights_gradient)
            v_b = 0.999 * v_b + 0.001 * np.square(bias_gradient)
            
            # Bias-corrected estimates
            m_w_hat = m_w / (1 - 0.9**t)
            m_b_hat = m_b / (1 - 0.9**t)
            v_w_hat = v_w / (1 - 0.999**t)
            v_b_hat = v_b / (1 - 0.999**t)
            
            # Update parameters
            self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            
            # Return updated optimizer state
            return input_gradient, (m_w, v_w, m_b, v_b, t)
        else:
            # Simple SGD update
            self.weights -= learning_rate * weights_gradient
            self.bias -= learning_rate * bias_gradient
            return input_gradient, None

class ReLU:
    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.maximum(0, inputs)
        
    def backward(self, gradient, learning_rate, optimizer_state=None):
        # ReLU gradient: 1 if input > 0, else 0
        return gradient * (self.inputs > 0), None

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.where(inputs > 0, inputs, inputs * self.alpha)
        
    def backward(self, gradient, learning_rate, optimizer_state=None):
        return gradient * np.where(self.inputs > 0, 1, self.alpha), None

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
    
    def backward(self, gradient, learning_rate, optimizer_state=None):
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
        if optimizer_state is not None:
            m_g, v_g, m_b, v_b, t = optimizer_state
            
            # Update biased first moment estimate
            m_g = 0.9 * m_g + 0.1 * dgamma
            m_b = 0.9 * m_b + 0.1 * dbeta
            
            # Update biased second raw moment estimate
            v_g = 0.999 * v_g + 0.001 * np.square(dgamma)
            v_b = 0.999 * v_b + 0.001 * np.square(dbeta)
            
            # Bias-corrected estimates
            m_g_hat = m_g / (1 - 0.9**t)
            m_b_hat = m_b / (1 - 0.9**t)
            v_g_hat = v_g / (1 - 0.999**t)
            v_b_hat = v_b / (1 - 0.999**t)
            
            # Update parameters
            self.gamma -= learning_rate * m_g_hat / (np.sqrt(v_g_hat) + 1e-8)
            self.beta -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            
            return dx, (m_g, v_g, m_b, v_b, t)
        else:
            self.gamma -= learning_rate * dgamma
            self.beta -= learning_rate * dbeta
            return dx, None

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
        
    def backward(self, gradient, learning_rate, optimizer_state=None):
        return gradient * self.mask, None

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.optimizer_states = []
        
    def add(self, layer):
        self.layers.append(layer)
        self.optimizer_states.append(None)
        
    def forward(self, inputs, training=True):
        for layer in self.layers:
            inputs = layer.forward(inputs, training)
        return inputs
    
    def backward(self, gradient, learning_rate):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            gradient, new_state = layer.backward(gradient, learning_rate, self.optimizer_states[i])
            if new_state is not None:
                self.optimizer_states[i] = new_state
        return gradient
    
    def init_optimizer(self):
        # Initialize Adam optimizer states for each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                # For fully connected layers
                m_w = np.zeros_like(layer.weights)
                v_w = np.zeros_like(layer.weights)
                m_b = np.zeros_like(layer.bias)
                v_b = np.zeros_like(layer.bias)
                self.optimizer_states[i] = (m_w, v_w, m_b, v_b, 1)
            elif isinstance(layer, BatchNormalization):
                # For batch normalization layers
                m_g = np.zeros_like(layer.gamma)
                v_g = np.zeros_like(layer.gamma)
                m_b = np.zeros_like(layer.beta)
                v_b = np.zeros_like(layer.beta)
                self.optimizer_states[i] = (m_g, v_g, m_b, v_b, 1)
    
    def train(self, X, y, epochs, batch_size, learning_rate_init=0.001, X_val=None, y_val=None, patience=10):
        n_samples = X.shape[0]
        history = {'loss': [], 'val_loss': []}
        
        # Initialize Adam optimizer
        self.init_optimizer()
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
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
                
                # Update Adam iteration counter
                for i, state in enumerate(self.optimizer_states):
                    if state is not None:
                        m_w, v_w, m_b, v_b, t = state
                        self.optimizer_states[i] = (m_w, v_w, m_b, v_b, t + 1)
            
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
                    # Save best weights (simplified version - in practice you'd deep copy all weights)
                    best_weights = []
                    for layer in self.layers:
                        if hasattr(layer, 'weights'):
                            best_weights.append((layer.weights.copy(), layer.bias.copy()))
                        else:
                            best_weights.append((None, None))
                else:
                    patience_counter += 1
                    if patience_counter >= patience and best_weights:  # Check if best_weights is not empty
                        print(f"Early stopping at epoch {epoch+1}")
                        # Restore best weights
                        for i, (weights, bias) in enumerate(best_weights):
                            if weights is not None:
                                self.layers[i].weights = weights
                            if bias is not None:
                                self.layers[i].bias = bias
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
            # Add small random noise to create a slightly different sample
            noise = np.random.normal(0, 0.02, size=X[i].shape)
            X_aug.append(X[i] + noise)
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

# --- MAIN EXECUTION ---
# Generate dataset
print("Generating dataset...")
X, y = generate_mixed_dataset(n_samples)

# Data augmentation
print("Augmenting data...")
X, y = augment_data(X, y, augmentation_factor=2)
print(f"Dataset size after augmentation: {X.shape[0]} samples")

# Normalize input features
print("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Further split training data to get validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Create neural network with improved architecture
print("Creating neural network...")
model = NeuralNetwork()
model.add(Layer(n_timesteps, 128, weight_decay=0.0001))  # Increased neurons, added L2 regularization
model.add(BatchNormalization(128))  # Added batch normalization
model.add(LeakyReLU(alpha=0.1))  # Using LeakyReLU instead of ReLU
model.add(Dropout(0.2))  # Added dropout for regularization
model.add(Layer(128, 64, weight_decay=0.0001))
model.add(BatchNormalization(64))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))
model.add(Layer(64, 2))  # Output: two frequency values

# Train model with improved training process
print("Training model...")
history = model.train(
    X_train, y_train,
    epochs=100,  # Increased epochs
    batch_size=128,  # Increased batch size
    learning_rate_init=0.001,
    X_val=X_val,
    y_val=y_val,
    patience=15  # Early stopping patience
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

# Print summary of improvements
print("\nSummary of Improvements:")
print("1. Increased dataset size from 10,000 to 20,000 samples")
print("2. Added data augmentation, doubling the effective dataset size")
print("3. Implemented Adam optimizer with learning rate scheduling")
print("4. Added batch normalization layers")
print("5. Replaced ReLU with LeakyReLU activation")
print("6. Added dropout layers for regularization")
print("7. Added L2 weight regularization")
print("8. Increased network capacity (more neurons)")
print("9. Implemented early stopping")
print("10. Increased training epochs and batch size")
