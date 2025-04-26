import numpy as np

# XOR function with binary input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Binary inputs
y = np.array([[0], [1], [1], [0]])               # Binary outputs

# Network parameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.5
epochs = 10000

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the network
for epoch in range(epochs):
    # Forward propagation
    # Hidden layer
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    # Output layer
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Back propagation
    # Output layer error
    error = y - a2
    delta2 = error * sigmoid_derivative(a2)
    
    # Hidden layer error
    delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(a1)
    
    # Update weights and biases
    W2 += np.dot(a1.T, delta2) * learning_rate
    b2 += np.sum(delta2, axis=0, keepdims=True) * learning_rate
    W1 += np.dot(X.T, delta1) * learning_rate
    b1 += np.sum(delta1, axis=0, keepdims=True) * learning_rate
    
    # Print progress
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Test the network
print("\nFinal predictions:")
for i in range(len(X)):
    # Forward pass to get predictions
    z1 = np.dot(X[i:i+1], W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    prediction = sigmoid(z2)
    
    # Convert to binary using threshold of 0.5
    binary_prediction = 1 if prediction > 0.5 else 0
    
    print(f"Input: {X[i]} -> Output: {binary_prediction} (Raw: {prediction[0][0]:.4f})")

# Verify XOR function
print("\nXOR Truth Table:")
print("A | B | A XOR B | Prediction")
print("--|---|---------|------------")
for i in range(len(X)):
    z1 = np.dot(X[i:i+1], W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    prediction = sigmoid(z2)
    binary_prediction = 1 if prediction > 0.5 else 0
    print(f"{int(X[i][0])} | {int(X[i][1])} | {int(y[i][0])}       | {binary_prediction}")