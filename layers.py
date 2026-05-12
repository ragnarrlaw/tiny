from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Set
from typing_extensions import override
import numpy as np
from numpy.typing import NDArray
from activations import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative


class Layer(ABC):
    units: int
    weights: NDArray[np.float32]
    bias: NDArray[np.float32]
    input_cache: NDArray[np.float32]
    z_cache: NDArray[np.float32]  # pre-activation cache
    a_cache: NDArray[np.float32]  # activation cache
    activation_func: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    activation_deriv_func: Callable[[NDArray[np.float32]], NDArray[np.float32]]

    def __init__(
        self,
        layer_type: str,
        name: str,
    ) -> None:
        self.layer_type: str = layer_type
        self.name: str = name

    def get_name(self) -> str:
        return self.name

    def get_weights(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return (self.weights, self.bias)

    @abstractmethod
    def get_units(self):
        return self.units

    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """returns inputs for the next layer"""
        pass

    @abstractmethod
    def backward(
        self,
        alpha: float,
        batch_size: int,
        error_signal: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """returns error signal for the previous layer"""
        pass

    @abstractmethod
    def init_params(self, no_input_features: int) -> None:
        """initializes the parameters"""
        pass


class InputLayer(Layer):
    def __init__(self, features: int, name="input") -> None:
        self.input_shape: int = features
        super().__init__(layer_type="input_layer", name=name)

    @override
    def get_units(self) -> int:
        return self.input_shape

    @override
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """returns inputs for the next layer"""
        return inputs

    @override
    def backward(
        self,
        alpha: float,
        batch_size: int,
        error_signal: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """returns error signal for the previous layer"""
        return error_signal

    @override
    def init_params(self, no_input_features: int) -> None:
        """initializes the parameters"""
        return None


class Dense(Layer):

    input_features: int = 0
    activation: str = ""

    def __init__(
        self,
        units: int,
        activation: str,
        name: str,
        bias: bool = True,
    ) -> None:
        super().__init__(layer_type="dense", name=name)
        self.activation = activation
        if activation == "relu":
            self.activation_func = relu
            self.activation_deriv_func = relu_derivative
        elif activation == "sigmoid":
            self.activation_func = sigmoid
            self.activation_deriv_func = sigmoid_derivative
        elif activation == "tanh":
            self.activation_func = tanh
            self.activation_deriv_func = tanh_derivative
        else:
            # linear
            self.activation_func = lambda z: z
            self.activation_deriv_func = lambda z: z
        self.units: int = units
        self.output_features: int = units
        self.add_bias: bool = bias

    @override
    def get_units(self) -> int:
        return self.units

    @override
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        self.input_cache = inputs
        self.z_cache = (inputs @ self.weights) + self.bias
        self.a_cache = self.activation_func(self.z_cache)
        return self.a_cache

    @override
    def backward(
        self,
        alpha: float,
        batch_size: int,
        error_signal: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """error_signal = dL/da from layer after"""
        if self.activation == "relu":
            dj_dz = self.activation_deriv_func(self.z_cache) * error_signal
        else:
            dj_dz = self.activation_deriv_func(self.a_cache) * error_signal

        dj_dw = self.input_cache.T @ dj_dz

        dj_db = None
        if self.add_bias:
            dj_db = dj_dz.sum(axis=0, keepdims=True)  # shape (1, units)
        else:
            dj_db = None

        error_to_prev = dj_dz @ self.weights.T

        self.weights = (self.weights - (alpha / batch_size) * dj_dw).astype(np.float32)

        if self.add_bias and dj_db is not None:
            self.bias = self.bias - (alpha / batch_size) * dj_db

        return error_to_prev.astype(np.float32)

    @override
    def init_params(self, no_input_features: int) -> None:
        self.input_features = no_input_features
        self.weights = (
            np.random.randn(self.input_features, self.output_features) * 0.1
        ).astype(np.float32)
        if self.add_bias:
            self.bias = np.zeros((1, self.output_features)).astype(np.float32)
        else:
            self.bias = np.zeros(1).astype(np.float32)


