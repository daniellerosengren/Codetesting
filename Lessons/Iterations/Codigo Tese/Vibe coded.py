import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt



# --- CONFIGURATION ---
n_samples = 10000       # Number of synthetic samples
n_timesteps = 20        # Length of time vector
t = np.linspace(0, 1, n_timesteps)
tau = 0.1               # Fixed interaction time
noise_level = 0.05      # Standard deviation of Gaussian noise

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




# Generate dataset
X, y = generate_mixed_dataset(n_samples)

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




# Define simple regression model
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_timesteps,)),
    Dense(64, activation='relu'),
    Dense(2)  # Output: two frequency values
])


# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=64)



# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MAE:", test_mae)

# Plot predictions vs true values for a few samples
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 4))
plt.plot(y_test[:50], label='True', marker='o')
plt.plot(y_pred[:50], label='Predicted', marker='x')
plt.title("Predicted vs True Frequencies")
plt.xlabel("Sample Index")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
