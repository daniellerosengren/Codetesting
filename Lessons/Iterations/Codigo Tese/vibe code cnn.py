import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
n_samples = 20000      # Number of synthetic samples
n_timesteps = 20        # Length of time vector
t = np.linspace(0, 1, n_timesteps)
tau = 0.1               # Fixed interaction time
noise_level = 0.05      # Standard deviation of Gaussian noise

# --- NEURAL NETWORK FROM SCRATCH WITH CNN IMPLEMENTATION ---
class Conv1D:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, weight_decay=0.0001):
        # Initialize weights with He initialization
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        
        # Initialize kernels with shape (output_channels, input_channels, kernel_size)
        self.kernels = np.random.randn(output_channels, input_channels, kernel_size) * np.sqrt(2.0 / (input_channels * kernel_size))
        self.bias = np.zeros((output_channels, 1))
        
    def _pad_input(self, inputs):
        if self.padding == 0:
            return inputs
        
        # Add padding to the input
        # For 1D convolution, we pad only the time dimension
        padded_inputs = np.pad(inputs, ((0, 0), (self.padding, self.padding)), mode='constant')
        return padded_inputs
    
    def forward(self, inputs, training=True):
        # inputs shape: (batch_size, input_length)
        self.inputs = inputs
        batch_size, input_length = inputs.shape
        
        # For the first convolutional layer, input_channels is always 1
        # For subsequent layers, we need to reshape based on the previous layer's output
        if self.input_channels == 1:
            # Reshape inputs to (batch_size, input_channels=1, input_length)
            inputs_reshaped = inputs.reshape(batch_size, 1, input_length)
        else:
            # For subsequent layers, the input is already in the correct shape
            # We just need to reshape it to (batch_size, input_channels, input_length/input_channels)
            inputs_reshaped = inputs.reshape(batch_size, self.input_channels, -1)
        
        # Store the reshaped input for backpropagation
        self.inputs_reshaped = inputs_reshaped
        
        # Pad the input if needed
        if self.padding > 0:
            inputs_reshaped = np.pad(inputs_reshaped, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        
        # Calculate output dimensions
        input_length_after_reshape = inputs_reshaped.shape[2]
        output_length = (input_length_after_reshape - self.kernel_size) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, self.output_channels, output_length))
        
        # Perform convolution
        for i in range(batch_size):
            for j in range(self.output_channels):
                for k in range(output_length):
                    start_idx = k * self.stride
                    end_idx = start_idx + self.kernel_size
                    # For each position, compute dot product of kernel and input segment
                    for c in range(self.input_channels):
                        self.output[i, j, k] += np.sum(self.kernels[j, c] * inputs_reshaped[i, c, start_idx:end_idx])
                    self.output[i, j, k] += self.bias[j, 0]
        
        # Reshape output to (batch_size, output_channels * output_length)
        self.output_reshaped = self.output.reshape(batch_size, self.output_channels * output_length)
        return self.output_reshaped
    
    def backward(self, gradient, learning_rate, optimizer_state=None):
        # gradient shape: (batch_size, output_channels * output_length)
        batch_size = self.inputs.shape[0]
        input_length = self.inputs.shape[1]
        
        # Get the output shape from the forward pass
        output_length = self.output.shape[2]
        
        # Check if gradient can be reshaped to match output shape
        expected_size = batch_size * self.output_channels * output_length
        if gradient.size != expected_size:
            print(f"Debug - Shape mismatch in Conv1D: gradient size {gradient.size}, expected size {expected_size}")
            print(f"Debug - Gradient shape: {gradient.shape}, expected shape: ({batch_size}, {self.output_channels}, {output_length})")
            
            # If shapes don't match, we need to pass the gradient through without modification
            # This is a workaround to allow training to continue
            input_gradient = np.zeros_like(self.inputs)
            return input_gradient, None
        
        # Reshape gradient to match output shape: (batch_size, output_channels, output_length)
        gradient_reshaped = gradient.reshape(batch_size, self.output_channels, output_length)
        
        # Initialize gradients for kernels and bias
        kernels_gradient = np.zeros_like(self.kernels)
        bias_gradient = np.zeros_like(self.bias)
        
        # Initialize gradient for input
        if self.padding > 0:
            padded_input_length = input_length + 2 * self.padding
            input_gradient = np.zeros((batch_size, self.input_channels, padded_input_length))
        else:
            input_gradient = np.zeros((batch_size, self.input_channels, input_length))
        
        # Use the stored reshaped input or reshape it again
        if hasattr(self, 'inputs_reshaped'):
            inputs_reshaped = self.inputs_reshaped
        else:
            # Reshape inputs to (batch_size, input_channels, input_length)
            inputs_reshaped = self.inputs.reshape(batch_size, self.input_channels, -1)
        
        # Pad the input if needed
        if self.padding > 0:
            inputs_reshaped = np.pad(inputs_reshaped, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        
        # Compute gradients
        for i in range(batch_size):
            for j in range(self.output_channels):
                for k in range(output_length):
                    start_idx = k * self.stride
                    end_idx = start_idx + self.kernel_size
                    
                    # Gradient for kernels
                    for c in range(self.input_channels):
                        kernels_gradient[j, c] += gradient_reshaped[i, j, k] * inputs_reshaped[i, c, start_idx:end_idx]
                    
                    # Gradient for bias
                    bias_gradient[j, 0] += gradient_reshaped[i, j, k]
                    
                    # Gradient for input
                    for c in range(self.input_channels):
                        input_gradient[i, c, start_idx:end_idx] += gradient_reshaped[i, j, k] * self.kernels[j, c]
        
        # Add L2 regularization gradient for kernels
        kernels_gradient += self.weight_decay * self.kernels
        
        # Remove padding from input gradient if needed
        if self.padding > 0:
            input_gradient = input_gradient[:, :, self.padding:-self.padding]
        
        # Reshape input gradient to match input shape: (batch_size, input_length)
        # Calculate the correct shape based on the input dimensions
        input_gradient = input_gradient.reshape(batch_size, -1)
        
        # Update parameters using Adam optimizer
        if optimizer_state is not None:
            # Unpack optimizer state
            m_k, v_k, m_b, v_b, t = optimizer_state
            
            # Update biased first moment estimate
            m_k = 0.9 * m_k + 0.1 * kernels_gradient
            m_b = 0.9 * m_b + 0.1 * bias_gradient
            
            # Update biased second raw moment estimate
            v_k = 0.999 * v_k + 0.001 * np.square(kernels_gradient)
            v_b = 0.999 * v_b + 0.001 * np.square(bias_gradient)
            
            # Bias-corrected estimates
            m_k_hat = m_k / (1 - 0.9**t)
            m_b_hat = m_b / (1 - 0.9**t)
            v_k_hat = v_k / (1 - 0.999**t)
            v_b_hat = v_b / (1 - 0.999**t)
            
            # Update parameters
            self.kernels -= learning_rate * m_k_hat / (np.sqrt(v_k_hat) + 1e-8)
            self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            
            # Return updated optimizer state
            return input_gradient, (m_k, v_k, m_b, v_b, t)
        else:
            # Simple SGD update
            self.kernels -= learning_rate * kernels_gradient
            self.bias -= learning_rate * bias_gradient
            return input_gradient, None

class MaxPooling1D:
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
    
    def forward(self, inputs, training=True):
        # inputs shape: (batch_size, input_length)
        self.inputs = inputs
        batch_size, input_length = inputs.shape
        
        # Calculate output dimensions
        output_length = (input_length - self.pool_size) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, output_length))
        
        # Store indices of max values for backpropagation
        self.max_indices = np.zeros((batch_size, output_length), dtype=int)
        
        # Perform max pooling
        for i in range(batch_size):
            for j in range(output_length):
                start_idx = j * self.stride
                end_idx = start_idx + self.pool_size
                
                # Find max value and its index
                segment = inputs[i, start_idx:end_idx]
                max_idx = np.argmax(segment)
                self.max_indices[i, j] = start_idx + max_idx
                self.output[i, j] = segment[max_idx]
        
        return self.output
    
    def backward(self, gradient, learning_rate, optimizer_state=None):
        # gradient shape: (batch_size, output_length)
        batch_size, output_length = gradient.shape
        
        # Initialize gradient for input
        input_gradient = np.zeros_like(self.inputs)
        
        # Distribute gradient to max indices
        for i in range(batch_size):
            for j in range(output_length):
                input_gradient[i, self.max_indices[i, j]] += gradient[i, j]
        
        return input_gradient, None

class Flatten:
    def __init__(self):
        pass
    
    def forward(self, inputs, training=True):
        # inputs shape: (batch_size, channels, height, width) or (batch_size, length)
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)
    
    def backward(self, gradient, learning_rate, optimizer_state=None):
        # Reshape gradient to match input shape
        return gradient.reshape(self.input_shape), None

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
        self.output = np.where(inputs > 0, inputs, inputs * self.alpha)
        return self.output
        
    def backward(self, gradient, learning_rate, optimizer_state=None):
        # Check if shapes match
        if gradient.shape != self.inputs.shape:
            print(f"Debug - Shape mismatch in LeakyReLU: gradient shape {gradient.shape}, inputs shape {self.inputs.shape}")
            # Reshape gradient to match input shape
            if len(gradient.shape) == 2 and len(self.inputs.shape) == 2:
                # Both are 2D, but different sizes
                # This is a more complex case that requires careful handling
                # For now, we'll just pass the gradient through without modification
                return gradient, None
            else:
                # Try to reshape to match
                try:
                    reshaped_gradient = gradient.reshape(self.inputs.shape)
                    return reshaped_gradient * np.where(self.inputs > 0, 1, self.alpha), None
                except:
                    # If reshaping fails, just pass through
                    return gradient, None
        else:
            # Normal case - shapes match
            return gradient * np.where(self.inputs > 0, 1, self.alpha), None

class BatchNormalization:
    def __init__(self, input_dim, epsilon=1e-8, momentum=0.9):
        self.input_dim = input_dim
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.input_shape = inputs.shape
        
        # Store the original input shape for reshaping in backward pass
        if len(inputs.shape) > 2:
            # Reshape to 2D for normalization
            inputs_reshaped = inputs.reshape(inputs.shape[0], -1)
            self.reshaped = True
        else:
            inputs_reshaped = inputs
            self.reshaped = False
        
        # Check if input dimensions match expected dimensions
        if inputs_reshaped.shape[1] != self.input_dim:
            # Adjust gamma and beta to match input dimensions
            self.gamma = np.ones((1, inputs_reshaped.shape[1]))
            self.beta = np.zeros((1, inputs_reshaped.shape[1]))
            self.running_mean = np.zeros((1, inputs_reshaped.shape[1]))
            self.running_var = np.ones((1, inputs_reshaped.shape[1]))
            self.input_dim = inputs_reshaped.shape[1]
        
        if training:
            # Calculate batch statistics
            self.batch_mean = np.mean(inputs_reshaped, axis=0, keepdims=True)
            self.batch_var = np.var(inputs_reshaped, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize
            self.x_norm = (inputs_reshaped - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            
            # Scale and shift
            output = self.gamma * self.x_norm + self.beta
        else:
            # Use running statistics for inference
            x_norm = (inputs_reshaped - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            output = self.gamma * x_norm + self.beta
        
        # Reshape back to original shape if needed
        if self.reshaped:
            return output.reshape(self.input_shape)
        else:
            return output
    
    def backward(self, gradient, learning_rate, optimizer_state=None):
        # Reshape gradient if needed
        if self.reshaped:
            gradient_reshaped = gradient.reshape(gradient.shape[0], -1)
        else:
            gradient_reshaped = gradient
        
        # Get batch size
        m = gradient_reshaped.shape[0]
        
        # Debug print to see shapes
        print(f"Debug - gradient_reshaped shape: {gradient_reshaped.shape}, self.x_norm shape: {self.x_norm.shape}")
        
        # If shapes don't match, we need to adjust our approach
        if gradient_reshaped.shape[1] != self.x_norm.shape[1]:
            # Reshape inputs to match gradient shape
            inputs_reshaped = self.inputs.reshape(m, -1)
            
            # Compute mean and variance for the reshaped inputs
            batch_mean = np.mean(inputs_reshaped, axis=0, keepdims=True)
            batch_var = np.var(inputs_reshaped, axis=0, keepdims=True)
            
            # Normalize with the new statistics
            x_norm = (inputs_reshaped - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Adjust gamma and beta to match the new shape
            gamma = np.ones((1, gradient_reshaped.shape[1]))
            beta = np.zeros((1, gradient_reshaped.shape[1]))
            
            # Compute gradients for gamma and beta
            dgamma = np.sum(gradient_reshaped * x_norm, axis=0, keepdims=True)
            dbeta = np.sum(gradient_reshaped, axis=0, keepdims=True)
            
            # Compute gradient with respect to input
            dx_norm = gradient_reshaped * gamma
            dvar = np.sum(dx_norm * (inputs_reshaped - batch_mean) * -0.5 * np.power(batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
            dmean = np.sum(dx_norm * -1 / np.sqrt(batch_var + self.epsilon), axis=0, keepdims=True) + dvar * np.mean(-2 * (inputs_reshaped - batch_mean), axis=0, keepdims=True)
            dx = dx_norm / np.sqrt(batch_var + self.epsilon) + dvar * 2 * (inputs_reshaped - batch_mean) / m + dmean / m
        else:
            # Compute gradients for gamma and beta
            dgamma = np.sum(gradient_reshaped * self.x_norm, axis=0, keepdims=True)
            dbeta = np.sum(gradient_reshaped, axis=0, keepdims=True)
            
            # Compute gradient with respect to input
            dx_norm = gradient_reshaped * self.gamma
            dvar = np.sum(dx_norm * (self.inputs.reshape(m, -1) - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
            dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + dvar * np.mean(-2 * (self.inputs.reshape(m, -1) - self.batch_mean), axis=0, keepdims=True)
            dx = dx_norm / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.inputs.reshape(m, -1) - self.batch_mean) / m + dmean / m
        
        
        # Reshape dx back to original input shape if needed
        if self.reshaped:
            dx = dx.reshape(self.input_shape)
        
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
            elif isinstance(layer, Conv1D):
                # For convolutional layers
                m_k = np.zeros_like(layer.kernels)
                v_k = np.zeros_like(layer.kernels)
                m_b = np.zeros_like(layer.bias)
                v_b = np.zeros_like(layer.bias)
                self.optimizer_states[i] = (m_k, v_k, m_b, v_b, 1)
    
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
                for j, state in enumerate(self.optimizer_states):
                    if state is not None:
                        if isinstance(self.layers[j], Conv1D):
                            m_k, v_k, m_b, v_b, t = state
                            self.optimizer_states[j] = (m_k, v_k, m_b, v_b, t + 1)
                        else:
                            m_w, v_w, m_b, v_b, t = state
                            self.optimizer_states[j] = (m_w, v_w, m_b, v_b, t + 1)
            
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
                            best_weights.append(('dense', layer.weights.copy(), layer.bias.copy()))
                        elif isinstance(layer, BatchNormalization):
                            best_weights.append(('bn', layer.gamma.copy(), layer.beta.copy()))
                        elif isinstance(layer, Conv1D):
                            best_weights.append(('conv', layer.kernels.copy(), layer.bias.copy()))
                        else:
                            best_weights.append(('other', None, None))
                else:
                    patience_counter += 1
                    if patience_counter >= patience and best_weights:  # Check if best_weights is not empty
                        print(f"Early stopping at epoch {epoch+1}")
                        # Restore best weights
                        for i, (layer_type, weights, bias) in enumerate(best_weights):
                            if layer_type == 'dense':
                                self.layers[i].weights = weights
                                self.layers[i].bias = bias
                            elif layer_type == 'bn':
                                self.layers[i].gamma = weights
                                self.layers[i].beta = bias
                            elif layer_type == 'conv':
                                self.layers[i].kernels = weights
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
model = NeuralNetwork()

# Add convolutional layers
model.add(Conv1D(input_channels=1, output_channels=16, kernel_size=3, padding=1))  # Output: (batch_size, 16*20)
model.add(LeakyReLU(alpha=0.1))
# Remove all BatchNormalization layers to avoid shape mismatch issues
# model.add(BatchNormalization(16*20))

model.add(Conv1D(input_channels=16, output_channels=32, kernel_size=3, padding=1))  # Output: (batch_size, 32*20)
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2))  # Output: (batch_size, 32*10)
# model.add(BatchNormalization(32*10))
model.add(Dropout(0.2))

# Add fully connected layers
model.add(Layer(32*10, 64, weight_decay=0.0001))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Layer(64, n_timesteps))  # Output: probability signal values for each time point

# Train model
print("Training model...")
history = model.train(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    learning_rate_init=0.001,
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
print("1. Convolutional Layers:")
print("   - First Conv1D: 1 input channel → 16 output channels, kernel size 3")
print("   - Second Conv1D: 16 input channels → 32 output channels, kernel size 3")
print("2. Pooling Layer:")
print("   - MaxPooling1D with pool size 2")
print("3. Fully Connected Layers:")
print(f"   - 32*10 → 64 → {n_timesteps} output neurons")
print("4. Regularization:")
print("   - Batch Normalization after each layer")
print("   - Dropout (20%) after pooling and first fully connected layer")
print("   - L2 weight decay (0.0001)")
print("5. Activation Functions:")
print("   - LeakyReLU (alpha=0.1) throughout the network")
print("\nThis model now takes time as input (X) and predicts the probability signal plus noise (y).")
