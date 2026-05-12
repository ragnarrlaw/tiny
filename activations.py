import numpy as np
from numpy.typing import NDArray

# these guys take either pre-activations or activations
def relu(z: NDArray[np.float32]) -> NDArray[np.float32]:
    """ReLU activation function "if z > 0 then z else 0"" - takes the pre-activations as input"""
    return np.maximum(0, z)


def relu_derivative(z: NDArray[np.float32]) -> NDArray[np.float32]:
    """ReLU activation's derivative function  - takes the pre-activations as input"""
    return (z > 0).astype(np.float32)


def sigmoid(z: NDArray[np.float32]) -> NDArray[np.float32]:
    """Sigmoid activation function "1 / (1 - e^(-z))" - takes the pre-activations as input"""
    return (1 / (1 + np.exp(-z).astype(np.float32))).astype(np.float32)


def sigmoid_derivative(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Sigmoid activation's derivative function " sigmoid(z) * (1 - sigmoid(z)) " - takes the activations as input"""
    return a * (1 - a)  # a = sigmoid(z)


def tanh(z: NDArray[np.float32]) -> NDArray[np.float32]:
    """Tanh activation function " tanh(z) " - takes the pre-activations as input"""
    return np.tanh(z)


def tanh_derivative(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Tanh activation's derivative function " 1 - tanh(z)^2 " - takes the activations as input"""
    return 1 - a**2  # a = tanh(z)


def softmax(z: NDArray[np.float32]) -> NDArray[np.float32]:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


