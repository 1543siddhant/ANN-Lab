#Write a Python program to plot a few activation functions that are being used in neural networks.

import numpy as np
import matplotlib.pyplot as plt

# Define the input range
x = np.linspace(-10, 10, 400)

# Define activation functions
# 1. Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# 2. Tanh
tanh = np.tanh(x)

# 3. ReLU
relu = np.maximum(0, x)


# Plotting the activation functions
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid, label='Sigmoid', color='blue')
plt.plot(x, tanh, label='Tanh', color='red')
plt.plot(x, relu, label='ReLU', color='green')


plt.title("Activation Functions in Neural Networks")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()