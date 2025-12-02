from catboost import CatBoostRegressor
from utils import get_data, evaluate_and_plot

# 1. Get Data
X_train, X_test, y_train, y_test, _ = get_data()

# 2. Define Model
model = CatBoostRegressor(iterations=100, verbose=0, random_state=42)

# 3. Train
print("Training CatBoost...")
model.fit(X_train, y_train)

# 4. Evaluate
evaluate_and_plot(model, X_test, y_test, "CatBoost")
