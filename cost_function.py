from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class CostFunction(ABC):
    @abstractmethod
    def cost(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
    ) -> float:
        pass

    @abstractmethod
    def backward(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        pass


class BinaryCrossEntropy(CostFunction):
    def cost(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
    ) -> float:
        return self.binary_cross_entropy(y_pred, y_true)

    def backward(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        return self.binary_cross_entropy_derivative(y_pred, y_true)

    @staticmethod
    def binary_cross_entropy(
        y_pred: NDArray[np.float32], y_true: NDArray[np.float32]
    ) -> float:
        """Binary Cross Entropy (with epsilon for numerical stability)"""
        eps = 1e-8
        return -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        ).astype(np.float64)

    @staticmethod
    def binary_cross_entropy_derivative(
        y_pred: NDArray[np.float32], y_true: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Derivative w.r.t. y_pred when using sigmoid (simplifies to y_pred - y_true)"""
        eps = 1e-8
        return (
            (y_pred - y_true) / (y_true.shape[0] * (y_pred * (1 - y_pred) + eps))
        ).astype(np.float32)


