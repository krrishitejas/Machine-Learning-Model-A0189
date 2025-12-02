from sklearn.neural_network import MLPRegressor
from utils import get_data, evaluate_and_plot

# 1. Get Data
X_train, X_test, y_train, y_test, _ = get_data()

# 2. Define Model
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)

# 3. Train
print("Training MLP...")
model.fit(X_train, y_train)

# 4. Evaluate
evaluate_and_plot(model, X_test, y_test, "MLP")
