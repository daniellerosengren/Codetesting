"""
CNN Implementation for Signal Splitting Using Keras
==================================================

This code implements a 1D Convolutional Neural Network (CNN) using Keras to process
time-series signals and split mixed signals into their component signals.
The implementation includes data generation, normalization, and a complete CNN architecture
with convolutional layers, pooling, and fully connected layers.
"""

import numpy as np
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
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


# Step 3: Visualize different normalization methods on a sample

print("Visualizing different normalization methods on a sample...")

sample_idx = 0
time = X[sample_idx].reshape(-1, 1)
signal = mixed[sample_idx].reshape(-1, 1)

scalers = [
    ("Raw Data", None),
    ("MaxAbsScaler", MaxAbsScaler()),
    ("StandardScaler", StandardScaler()),
    ("PowerTransformer", PowerTransformer()),
    ("MinMaxScaler", MinMaxScaler()),
    ("RobustScaler", RobustScaler()),
    ("QuantileTransformer", QuantileTransformer(output_distribution='uniform'))
]

plt.figure(figsize=(18, 12))
for i, (name, scaler) in enumerate(scalers):
    plt.subplot(3, 3, i+1)
    if scaler is None:
        plt.scatter(time, signal, alpha=0.7)
        plt.title("Raw Data")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
    else:
        scaled_signal = scaler.fit_transform(signal)
        plt.scatter(time, scaled_signal, alpha=0.7)
        plt.title(name)
        plt.xlabel("Time")
        plt.ylabel("Scaled Amplitude")
    plt.grid()

plt.tight_layout()
plt.show()

