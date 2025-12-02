import lightgbm as lgb
from utils import get_data, evaluate_and_plot

# 1. Get Data
X_train, X_test, y_train, y_test, _ = get_data()

# 2. Define Model
model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

# 3. Train
print("Training LightGBM...")
model.fit(X_train, y_train)

# 4. Evaluate
evaluate_and_plot(model, X_test, y_test, "LightGBM")
