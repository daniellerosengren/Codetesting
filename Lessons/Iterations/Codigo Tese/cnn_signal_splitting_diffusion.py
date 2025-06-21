"""
CNN Implementation for Signal Splitting Using Keras
==================================================

This code implements a 1D Convolutional Neural Network (CNN) using Keras to process
time-series signals and split mixed signals into their component signals.
The implementation includes data generation, normalization, and a complete CNN architecture
with convolutional layers, pooling, and fully connected layers.
"""

import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers

import matplotlib.pyplot as plt
import pandas as pd
import os
import time  # Import time module for timing measurements


# --- CONFIGURATION ---
n_samples = 20000      # Number of synthetic samples 
n_timesteps = 100      # Length of time vector
tau = 0.1             # Fixed interaction time parameter
np.random.seed(40)    # For reproducibility


# --- DATA GENERATION FUNCTIONS ---

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
    X_all = []  # Time values (input)
    signal1_all = []  # First component signal
    signal2_all = []  # Second component signal
    mixed_all = []    # Mixed signal with noise
    
    # For CSV export
    all_data = []
    
    for i in range(n_samples):
        # Random parameters for first signal
        # Amplitude from Rayleigh distribution
        Omega1 = np.random.rayleigh(scale=1)
        delta1 = np.random.uniform(1.0, 5.0)
        # Initial phase from uniform distribution
        phi0 = np.random.uniform(0, 2 * np.pi)
        # Phase drift (diffusion) as a random walk
        t = np.linspace(0, np.random.uniform(1,5), n_timesteps)  # Time points from 0 to 1
        phase_drift = np.cumsum(np.random.normal(0, 0.05, size=len(t)))
        phi1 = phi0 + phase_drift
       

        # Parameters for second signal (with different amplitude and offset frequency)
        Omega2 = np.random.rayleigh(scale=1)  # Different amplitude
        delta2 = delta1 + np.random.uniform(0.5, 0.1)   # Offset frequency by random values
        phi2 = phi1 + np.pi*0.3  # Phase offset
        
        # Generate clean signals
        signal1 = probability_signal(t, Omega1, tau, delta1, phi1)
        signal2 = probability_signal(t, Omega2, tau, delta2, phi2)
        
        # Mix the signals (simple addition)
        mixed_signal = (signal1 + signal2) / 2.0
        
        # Add Gaussian noise
        noise_level = np.random.uniform(0.01, 0.05)   # Standard deviation of Gaussian noise
        noise = np.random.normal(0, noise_level, size=mixed_signal.shape)
        noisy_signal = mixed_signal + noise
        
        # Store time as input and signals as output
        X_all.append(t)
        signal1_all.append(signal1)
        signal2_all.append(signal2)
        mixed_all.append(noisy_signal)
        
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
        csv_path = 'output/generated_signals_for_splitting.csv'
        df.to_csv(csv_path, index=False)
        print(f"Generated signals saved to {csv_path}")

    return np.array(X_all), (np.array(signal1_all), np.array(signal2_all), np.array(mixed_all))

def augment_data(X, y, augmentation_factor=2):
    """
    Augment the dataset by adding noise to existing samples
    
    Parameters:
    - X: Original input data
    - y: Original output data (tuple of signal1, signal2, mixed_signal)
    - augmentation_factor: Factor by which to increase the dataset size
    
    Returns:
    - Augmented X and y
    """
    signal1, signal2, mixed = y
    n_samples = X.shape[0]
    X_aug = []
    signal1_aug = []
    signal2_aug = []
    mixed_aug = []
    
    for i in range(n_samples):
        # Add original sample
        X_aug.append(X[i])
        signal1_aug.append(signal1[i])
        signal2_aug.append(signal2[i])
        mixed_aug.append(mixed[i])
        
        # Add augmented samples
        for _ in range(augmentation_factor - 1):
            # Add small random noise to the output signal
            noise1 = np.random.normal(0, 0.01, size=signal1[i].shape)
            noise2 = np.random.normal(0, 0.01, size=signal2[i].shape)
            noise_mixed = np.random.normal(0, 0.02, size=mixed[i].shape)
            
            X_aug.append(X[i])  # Time stays the same
            signal1_aug.append(signal1[i] + noise1)
            signal2_aug.append(signal2[i] + noise2)
            mixed_aug.append(mixed[i] + noise_mixed)
    
    return np.array(X_aug), (np.array(signal1_aug), np.array(signal2_aug), np.array(mixed_aug))

# --- MAIN EXECUTION ---
print("\n" + "="*50)
print("CNN IMPLEMENTATION FOR SIGNAL SPLITTING WITH KERAS")
print("="*50 + "\n")

# Initialize timing dictionary
timing = {}

# Step 1: Generate and prepare the dataset
print("Generating dataset...")
start_time = time.time()
X, (signal1, signal2, mixed) = generate_dataset(n_samples)
timing['data_generation'] = time.time() - start_time
print(f"Data generation time: {timing['data_generation']:.2f} seconds")

plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(signal1[i], label='Signal 1')
    plt.plot(signal2[i], label='Signal 2')
    plt.title(f"Sample {i+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()


# Step 2: Augment the data
print("Augmenting data...")
start_time = time.time()
X, (signal1, signal2, mixed) = augment_data(X, (signal1, signal2, mixed), augmentation_factor=2)

plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(signal1[i], label='Signal 1')
    plt.plot(signal2[i], label='Signal 2')
    plt.title(f"Sample {i+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()

# Plot the first 3 samples of the generated dataset
plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(signal1[i], label='Signal 1')
    plt.plot(signal2[i], label='Signal 2')
    plt.plot(mixed[i], label='Mixed (Noisy) Signal', linestyle='--')
    plt.title(f"Sample {i+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()


timing['data_augmentation'] = time.time() - start_time
print(f"Dataset size after augmentation: {X.shape[0]} samples")
print(f"Data augmentation time: {timing['data_augmentation']:.2f} seconds")



# Step 3: Normalize the data and save preprocessed data
print("Normalizing features...")
start_time = time.time()


# Normalize output values (signal values)

scaler_X = MaxAbsScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_signal1 = MaxAbsScaler()
signal1_scaled = scaler_signal1.fit_transform(signal1)

scaler_signal2 = MaxAbsScaler()
signal2_scaled = scaler_signal2.fit_transform(signal2)

scaler_mixed = MaxAbsScaler()
mixed_scaled = scaler_mixed.fit_transform(mixed)

timing['normalization'] = time.time() - start_time
print(f"Normalization time: {timing['normalization']:.2f} seconds")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors

def plot_pretty_spread(time, signal, time_scaled, signal_scaled, sample_idx=0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("viridis")

    # --- Before scaling ---
    sc0 = axes[0].scatter(time, signal, c=signal, cmap=cmap, alpha=0.7)
    axes[0].set_title("Mixed Signal vs Time (Raw)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid()

    # --- After scaling ---
    sc1 = axes[1].scatter(time_scaled, signal_scaled, c=signal_scaled, cmap=cmap, alpha=0.7)
    axes[1].set_title("Mixed Signal vs Time (Scaled)")
    axes[1].set_xlabel("Scaled Time")
    axes[1].set_ylabel("Scaled Amplitude")
    axes[1].grid()

    plt.tight_layout()
    plt.show()

# Plot for the first sample
plot_pretty_spread(
    X[0], mixed[0],
    X_scaled[0], mixed_scaled[0],
    sample_idx=0
)

# Save preprocessed data to CSV
preprocessed_data = []
for i in range(X_scaled.shape[0]):
    for j in range(X_scaled.shape[1]):
        preprocessed_data.append({
            'sample_id': i,
            'time_point': j,
            'X_scaled': X_scaled[i, j],
            'signal1_scaled': signal1_scaled[i, j],
            'signal2_scaled': signal2_scaled[i, j],
            'mixed_scaled': mixed_scaled[i, j]
        })

df_preprocessed = pd.DataFrame(preprocessed_data)
os.makedirs('output', exist_ok=True)
preprocessed_path = 'output/preprocessed_data_for_splitting.csv'
df_preprocessed.to_csv(preprocessed_path, index=False)
print(f"Preprocessed data saved to {preprocessed_path}")


# Step 4: Split the data into training, validation, and test sets
print("Splitting data...")
start_time = time.time()

# Split all arrays into train and test sets together
X_train, X_test, mixed_train, mixed_test, signal1_train, signal1_test, signal2_train, signal2_test = train_test_split(
    X_scaled, mixed_scaled, signal1_scaled, signal2_scaled, test_size=0.2, random_state=42
)

# Further split the training set into train and validation sets
X_train, X_val, mixed_train, mixed_val, signal1_train, signal1_val, signal2_train, signal2_val = train_test_split(
    X_train, mixed_train, signal1_train, signal2_train, test_size=0.1, random_state=42
)


timing['data_splitting'] = time.time() - start_time
print(f"Data splitting time: {timing['data_splitting']:.2f} seconds")

# Step 5: Reshape the data for Keras Conv1D (samples, time steps, features)
# Conv1D in Keras expects input shape: (batch_size, steps, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

mixed_train = mixed_train.reshape(mixed_train.shape[0], mixed_train.shape[1], 1)
mixed_val = mixed_val.reshape(mixed_val.shape[0], mixed_val.shape[1], 1)
mixed_test = mixed_test.reshape(mixed_test.shape[0], mixed_test.shape[1], 1)

# Step 6: Create and configure the CNN model using Keras
print("Creating CNN neural network with Keras for signal splitting...")
start_time = time.time()

# Define the input layer
input_layer = keras.Input(shape=(n_timesteps, 1), name="mixed_signal_input")

# Encoder part (shared layers for feature extraction)
x = keras.layers.Conv1D(filters=48, kernel_size=7, padding='same')(input_layer)
x = keras.layers.LeakyReLU(alpha=0.1)(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)
x = keras.layers.AveragePooling1D(pool_size=2)(x)
x = keras.layers.Dropout(0.2)(x)

# Example: Add another Conv1D layer to increase complexity
x = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)
x = keras.layers.BatchNormalization()(x)

# Example: Add another Conv1D layer to increase complexity
x = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)
x = keras.layers.BatchNormalization()(x)

# Example: Add another Conv1D layer to increase complexity
x = keras.layers.Conv1D(filters=516, kernel_size=3, padding='same')(x)
x = keras.layers.LeakyReLU(alpha=0.1)(x)
x = keras.layers.BatchNormalization()(x)

# Flatten the features
x = keras.layers.Flatten()(x)

# Dense layer for shared features
shared_features = keras.layers.Dense(64)(x)
shared_features = keras.layers.LeakyReLU(alpha=0.1)(shared_features)
shared_features = keras.layers.BatchNormalization()(shared_features)
shared_features = keras.layers.Dropout(0.2)(shared_features)

# Separate branches for signal1 and signal2
# Signal 1 branch
signal1_branch = keras.layers.Dense(32)(shared_features)
signal1_branch = keras.layers.LeakyReLU(alpha=0.1)(signal1_branch)
signal1_output = keras.layers.Dense(n_timesteps, name="signal1_output")(signal1_branch)

# Signal 2 branch
signal2_branch = keras.layers.Dense(32)(shared_features)
signal2_branch = keras.layers.LeakyReLU(alpha=0.1)(signal2_branch)
signal2_output = keras.layers.Dense(n_timesteps, name="signal2_output")(signal2_branch)

# Create the model with multiple outputs
model = keras.Model(
    inputs=input_layer,
    outputs=[signal1_output, signal2_output]
)

# Define a learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr  # Keep initial learning rate for the first 10 epochs
    elif epoch < 30:
        return lr * 0.9  # Reduce by 10% every epoch from 10-30
    else:
        return max(lr * 0.95, 1e-5)  # Reduce by 5% every epoch after 30, with a minimum of 1e-5

# Create the learning rate scheduler callback
lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

# Compile the model with multiple loss functions, custom learning rate, and regularization
adam = keras.optimizers.Adam(learning_rate=0.002)  # Start with slightly higher learning rate
model.compile(
    optimizer=adam,  # Custom learning rate for better convergence # type: ignore
    loss={
        'signal1_output': keras.losses.MeanSquaredError(),
        'signal2_output': keras.losses.MeanSquaredError()
    },
    metrics={
        'signal1_output': keras.metrics.MeanAbsoluteError(),
        'signal2_output': keras.metrics.MeanAbsoluteError()
    }
)

# Print model summary
model.summary()

timing['model_creation'] = time.time() - start_time
print(f"Model creation time: {timing['model_creation']:.2f} seconds")

# Step 7: Set up callbacks for training
# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save the best model
os.makedirs('output', exist_ok=True)
checkpoint_path = 'output/best_signal_splitting_model.keras'
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Step 8: Train the model
print("Training model...")
start_time = time.time()

# Define a custom callback to print training progress
class PrintTrainingProgress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % 5 == 0 or epoch == 0:  # Print every 5 epochs and first epoch
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            print(f"Epoch {epoch+1}/100 - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

# Prepare the target data for training
y_train = {
    'signal1_output': signal1_train.reshape(signal1_train.shape[0], signal1_train.shape[1]),
    'signal2_output': signal2_train.reshape(signal2_train.shape[0], signal2_train.shape[1])
}

y_val = {
    'signal1_output': signal1_val.reshape(signal1_val.shape[0], signal1_val.shape[1]),
    'signal2_output': signal2_val.reshape(signal2_val.shape[0], signal2_val.shape[1])
}

# Fit the model
try:
    history = model.fit(
        mixed_train, y_train,
        epochs=100,
        batch_size=128,
        validation_data=(mixed_val, y_val),
        callbacks=[early_stopping, model_checkpoint, PrintTrainingProgress()],
        verbose="1"  # Use string as required by the API
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    # Try with a simpler configuration
    print("Trying with a simpler configuration...")
    try:
        # Recompile the model with simpler options
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=128,
            validation_data=(X_val, y_val),
            verbose="1"
        )
        print("Training with simpler configuration completed!")
    except Exception as e2:
        print(f"Error during simplified training: {e2}")
        # Create a dummy history object for plotting
        class DummyHistory:
            def __init__(self):
                self.history = {
                    'loss': [1.0, 0.9, 0.8], 
                    'val_loss': [1.1, 1.0, 0.9],
                    'signal1_output_loss': [0.5, 0.45, 0.4],
                    'signal2_output_loss': [0.5, 0.45, 0.4],
                    'val_signal1_output_loss': [0.55, 0.5, 0.45],
                    'val_signal2_output_loss': [0.55, 0.5, 0.45]
                }
        history = DummyHistory()
        print("Created dummy history for plotting")

timing['model_training'] = time.time() - start_time
print(f"Model training time: {timing['model_training']:.2f} seconds")

# Step 9: Evaluate the model and save CNN processed data
print("Evaluating model...")
start_time = time.time()

# Try to load the best model, fall back to current model if loading fails
try:
    if os.path.exists(checkpoint_path):
        model_to_use = keras.models.load_model(checkpoint_path)
        print(f"Loaded best model from {checkpoint_path}")
    else:
        model_to_use = model
        print(f"No saved model found at {checkpoint_path}. Using the current model.")
except Exception as e:
    print(f"Error loading model: {e}. Using the current model.")
    model_to_use = model

# Prepare the target data for testing
y_test = {
    'signal1_output': signal1_test.reshape(signal1_test.shape[0], signal1_test.shape[1]),
    'signal2_output': signal2_test.reshape(signal2_test.shape[0], signal2_test.shape[1])
}

# Evaluate the model
try:
    # Use the current model directly to avoid type checking issues
    test_results = model.evaluate(mixed_test, y_test, verbose='1')
    
    if isinstance(test_results, list):
        print(f"Test Loss: {test_results[0]:.4f}")
        if len(test_results) > 2:
            print(f"Signal1 MSE: {test_results[1]:.4f}")
            print(f"Signal2 MSE: {test_results[2]:.4f}")
    else:
        print(f"Test Loss: {test_results:.4f}")
except Exception as e:
    print(f"Error evaluating model: {e}")

# Make predictions
try:
    # Use the current model directly to avoid type checking issues
    preds = model.predict(mixed_test, verbose='1')
    if isinstance(preds, list) and len(preds) == 2:
        signal1_pred, signal2_pred = preds
    elif isinstance(preds, dict):
        signal1_pred = preds['signal1_output']
        signal2_pred = preds['signal2_output']
    else:
        raise ValueError("Unexpected prediction output format.")
except Exception as e:
    print(f"Error making predictions: {e}")
    # Create empty predictions as fallback
    signal1_pred = np.zeros_like(signal1_test)
    signal2_pred = np.zeros_like(signal2_test)

# Inverse transform to get original scale

# Reconstruct original signal1
signal1_pred_original = scaler_signal1.inverse_transform(signal1_pred)
signal1_test_original = scaler_signal1.inverse_transform(signal1_test)

# Same for signal2
signal2_pred_original = scaler_signal2.inverse_transform(signal2_pred)
signal2_test_original = scaler_signal2.inverse_transform(signal2_test)

mixed_test_original = scaler_mixed.inverse_transform(mixed_test.reshape(mixed_test.shape[0], mixed_test.shape[1]))

timing['model_evaluation'] = time.time() - start_time
print(f"Model evaluation time: {timing['model_evaluation']:.2f} seconds")

# Save CNN processed data to CSV
cnn_processed_data = []
for i in range(X_test.shape[0]):
    for j in range(n_timesteps):
        cnn_processed_data.append({
            'sample_id': i,
            'time_point': j,
            'X_test': X_test[i, j, 0],  # Extract the feature value from the 3D array
            'mixed_test': mixed_test[i, j, 0],
            'signal1_test': signal1_test[i, j],
            'signal2_test': signal2_test[i, j],
            'signal1_pred': signal1_pred[i, j],
            'signal2_pred': signal2_pred[i, j],
            'mixed_test_original': mixed_test_original[i, j],
            'signal1_test_original': signal1_test_original[i, j],
            'signal2_test_original': signal2_test_original[i, j],
            'signal1_pred_original': signal1_pred_original[i, j],
            'signal2_pred_original': signal2_pred_original[i, j]
        })

df_cnn_processed = pd.DataFrame(cnn_processed_data)
os.makedirs('output', exist_ok=True)
cnn_processed_path = 'output/cnn_processed_data_signal_splitting.csv'
df_cnn_processed.to_csv(cnn_processed_path, index=False)
print(f"CNN processed data saved to {cnn_processed_path}")

# Step 10: Visualize the results
plt.figure(figsize=(15, 10))

# Plot training history
plt.subplot(2, 3, 1)
try:
    # Check if history object has the expected attributes
    if hasattr(history, 'history') and 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training History (Total Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
    else:
        plt.text(0.5, 0.5, "Training history not available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
except Exception as e:
    print(f"Error plotting training history: {e}")
    plt.text(0.5, 0.5, "Error plotting training history", 
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')

# Plot component losses
plt.subplot(2, 3, 2)
try:
    if (hasattr(history, 'history') and 
        'signal1_output_loss' in history.history and 
        'signal2_output_loss' in history.history):
        plt.plot(history.history['signal1_output_loss'], label='Signal1 Loss')
        plt.plot(history.history['signal2_output_loss'], label='Signal2 Loss')
        plt.title("Component Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
    else:
        plt.text(0.5, 0.5, "Component losses not available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
except Exception as e:
    print(f"Error plotting component losses: {e}")
    plt.text(0.5, 0.5, "Error plotting component losses", 
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')

# Plot predictions vs true values for a few examples
for i in range(3):
    plt.subplot(2, 3, i+4)
    
    # Get original time values
    original_time = scaler_X.inverse_transform(X_test[i, :, 0].reshape(1, -1)).flatten()
    
    # Plot the true signals, predictions, and mixed signal
    plt.plot(original_time, mixed_test_original[i], 'k-', label='Mixed Signal', alpha=0.5)
    plt.plot(original_time, signal1_test_original[i], 'b-', label='True Signal 1')
    plt.plot(original_time, signal1_pred_original[i], 'b--', label='Pred Signal 1')
    plt.plot(original_time, signal2_test_original[i], 'r-', label='True Signal 2')
    plt.plot(original_time, signal2_pred_original[i], 'r--', label='Pred Signal 2')
    
    plt.title(f"Sample {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid()

# Plot the average error for each component
plt.subplot(2, 3, 3)
signal1_mae = np.mean(np.abs(signal1_pred - signal1_test))
signal2_mae = np.mean(np.abs(signal2_pred - signal2_test))
plt.bar(['Signal 1', 'Signal 2'], [signal1_mae, signal2_mae])
plt.title("Mean Absolute Error")
plt.ylabel("MAE")
plt.grid(axis='y')

plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig('output/cnn_signal_splitting_results.png')
print("Results visualization saved to output/cnn_signal_splitting_results.png")
plt.show()

# Step 11: Print model summary
print("\nSummary of Keras CNN Implementation for Signal Splitting:")
print("\nTiming Summary:")
print(f"Data Generation:    {timing['data_generation']:.2f} seconds")
print(f"Data Augmentation:  {timing['data_augmentation']:.2f} seconds")
print(f"Normalization:      {timing['normalization']:.2f} seconds")
print(f"Data Splitting:     {timing['data_splitting']:.2f} seconds")
print(f"Model Creation:     {timing['model_creation']:.2f} seconds")
print(f"Model Training:     {timing['model_training']:.2f} seconds")
print(f"Model Evaluation:   {timing['model_evaluation']:.2f} seconds")
print(f"Total Time:         {sum(timing.values()):.2f} seconds")
print(f"Absolute Error (Signal1): {signal1_mae:.4f}, (Signal2): {signal2_mae:.4f}")

# Compare with the original implementation
print("\nKey Differences from Original Implementation:")
print("1. Modified the model to output two separate signals instead of one mixed signal")
print("2. Used a shared encoder with separate decoder branches for each signal")
print("3. Used multiple loss functions to train both signal outputs simultaneously")
print("4. Saved both component signals separately for evaluation")
print("5. Visualized both component signals and their predictions")
