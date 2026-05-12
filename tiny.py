from dgen import make_coffee_data
from network import Network
from layers import InputLayer, Dense
from cost_function import BinaryCrossEntropy

x_train, y_train = make_coffee_data()
x_train_norm = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)

print(f"Some of x_train: {x_train[10::]} and Some of y_train: {y_train[10:]}")

nn = Network(
    [
        InputLayer(2, name="input"),
        Dense(8, activation="relu", name="l1"),
        Dense(8, activation="relu", name="l2"),
        Dense(1, activation="sigmoid", name="output"),
    ]
)

nn.compile(cost_function=BinaryCrossEntropy())
nn.fit(x_train_norm, y_train, learning_rate=0.01, epochs=10000)
y_pred = nn.predict(x_train_norm)
