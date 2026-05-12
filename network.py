import numpy as np
from numpy.typing import NDArray
from cost_function import CostFunction
from typing import Dict, List, Set
from layers import InputLayer, Dense, Layer


class Network:
    cost_func: CostFunction
    layer_index: Dict[str, Layer] = dict()

    def __init__(
        self,
        layers: List[InputLayer | Dense],
    ) -> None:
        self.layers: List[InputLayer | Dense] = layers

    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        current: NDArray[np.float32] = x
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def compile(
        self,
        cost_function: CostFunction,
    ):
        self.cost_func = cost_function
        # check for duplicate names
        tmp_names: Set[str] = set()
        output_features = 0
        for layer in self.layers:
            if layer.get_name() in tmp_names:
                raise NameError(f"Duplicate layer names: {layer.get_name()}")
            else:
                tmp_names.add(layer.get_name())
            layer.init_params(output_features)
            output_features = layer.get_units()
            self.layer_index[layer.get_name()] = layer

    def fit(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.float32],
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 1000,
    ) -> None:
        n_samples = x.shape[0]
        num_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            epoch_loss = 0.0

            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, n_samples)

                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                current = x_batch
                for layer in self.layers:
                    current = layer.forward(current)

                y_pred = current

                loss = self.cost_func.cost(y_pred, y_batch)
                epoch_loss += loss * len(x_batch)

                gradient = self.cost_func.backward(y_true=y_batch, y_pred=y_pred)

                for layer in reversed(self.layers):
                    gradient = layer.backward(
                        alpha=learning_rate,
                        batch_size=len(x_batch),
                        error_signal=gradient,
                    )
            print(f"epoch_loss: {epoch_loss}")

            avg_loss = epoch_loss / n_samples
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f}")



