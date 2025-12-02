import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from utils import get_data, evaluate_and_plot

# 1. Get Data
X_train, X_test, y_train, y_test, _ = get_data()

# Reshape for GRU: (samples, time steps, features)
X_train_gru = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_gru = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 2. Define Model
model = Sequential()
model.add(Input(shape=(1, X_train.shape[1])))
model.add(GRU(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 3. Train
print("Training GRU...")
model.fit(X_train_gru, y_train, epochs=100, batch_size=32, verbose=0)

# 4. Evaluate
evaluate_and_plot(model, X_test_gru, y_test, "GRU")
