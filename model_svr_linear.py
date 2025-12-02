from sklearn.svm import SVR
from utils import get_data, evaluate_and_plot

# 1. Get Data
X_train, X_test, y_train, y_test, _ = get_data()

# 2. Define Model
model = SVR(kernel='linear', C=10)

# 3. Train
print("Training SVR (Linear)...")
model.fit(X_train, y_train)

# 4. Evaluate
evaluate_and_plot(model, X_test, y_test, "SVR (Linear)")
