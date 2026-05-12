import numpy as np

# mainly for data generation purposes

# House Price Prediction
def make_house_data(seed=42):
    """Predicting house price (in $1000s) from size (sqft) and number of bedrooms"""
    np.random.seed(seed)
    n = 300
    size = np.random.uniform(800, 3500, n)  # sqft
    rooms = np.random.randint(2, 6, n)  # bedrooms
    X = np.column_stack((size, rooms))
    # True price = 0.3*size + 15*rooms + noise + base
    y = 0.3 * size + 15 * rooms + np.random.normal(0, 20, n) + 50
    y = y.reshape(-1, 1) / 1000  # price in $1000s
    return X.astype(np.float32), y.astype(np.float32)


# Coffee Roasting
def make_coffee_data(seed=2):
    """Predict whether the given temperature and the duration is good for coffe roasting"""
    np.random.seed(seed)
    n = 400
    X = np.random.rand(n, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # duration 11.5-15.5
    X[:, 0] = X[:, 0] * 110 + 150  # temp 150-260 °C
    y = np.zeros(n)
    for i, (t, d) in enumerate(X):
        if 175 < t < 260 and 12 < d < 15:
            y[i] = 1
    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)


# Heart Disease Prediction
def make_heart_data(seed=7):
    """Predict whether a patient has heart disease risk based on age and cholesterol"""
    np.random.seed(seed)
    n = 500
    age = np.random.uniform(30, 80, n)
    chol = np.random.uniform(120, 350, n)
    X = np.column_stack((age, chol))
    # Simple rule + noise
    y = ((age > 50) & (chol > 220)).astype(float)
    noise = np.random.rand(n) < 0.15
    y[noise] = 1 - y[noise]
    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)


# Iris Flower Species
def make_iris_data(seed=42):
    """Classify iris flower species from sepal/petal measurements"""
    np.random.seed(seed)
    n = 300
    # 3 clusters
    centers = np.array(
        [[5.0, 3.5, 1.5, 0.2], [6.0, 3.0, 4.5, 1.5], [6.5, 3.0, 5.5, 2.0]]
    )
    X = np.vstack([centers[i] + np.random.randn(n // 3, 4) * 0.4 for i in range(3)])
    y = np.repeat([0, 1, 2], n // 3).reshape(-1, 1)
    return X.astype(np.float32), y.astype(np.float32)
