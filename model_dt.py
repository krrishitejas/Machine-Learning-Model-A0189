from sklearn.tree import DecisionTreeRegressor
from utils import get_data, evaluate_and_plot

# 1. Get Data
X_train, X_test, y_train, y_test, _ = get_data()

# 2. Define Model
model = DecisionTreeRegressor(random_state=42)

# 3. Train
print("Training Decision Tree...")
model.fit(X_train, y_train)

# 4. Evaluate
evaluate_and_plot(model, X_test, y_test, "Decision Tree")
