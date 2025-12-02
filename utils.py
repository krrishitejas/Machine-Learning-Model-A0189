import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

def get_data(file_path='2010 to 2019.xlsx'):
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

    # Preprocessing
    if 'WATER LEVEL (M)' in df.columns:
        df['WATER LEVEL (M)'] = pd.to_numeric(df['WATER LEVEL (M)'], errors='coerce')

    # Drop rows where target 'Seepage (l/day)' is NaN
    df = df.dropna(subset=['Seepage (l/day)'])

    # Fill missing values in features with mean
    df = df.fillna(df.mean())

    # Outlier Removal: Remove top 5% of Seepage values
    threshold = df['Seepage (l/day)'].quantile(0.95)
    df = df[df['Seepage (l/day)'] <= threshold]

    # Feature Engineering: Lag Features (Window = 3)
    features_to_lag = df.columns.tolist()
    for col in features_to_lag:
        for lag in range(1, 4):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    df = df.dropna()
    
    X = df.drop('Seepage (l/day)', axis=1)
    y = df['Seepage (l/day)']

    # Log Transform Target
    y_log = np.log1p(y)

    # Split Data
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_log, y_test_log, scaler

def evaluate_and_plot(model, X_test, y_test_log, model_name, scaler=None):
    # Predict
    y_pred_log = model.predict(X_test)
    
    # For GRU/RNN, output might need flattening
    if len(y_pred_log.shape) > 1:
        y_pred_log = y_pred_log.flatten()

    # Transform back to original scale
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)
    
    # Metrics
    r2_log = r2_score(y_test_log, y_pred_log)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n--- {model_name} ---")
    print(f"Log R^2: {r2_log:.4f}")
    print(f"Original R^2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.savefig(f'plot_{model_name.replace(" ", "_")}.png')
    plt.close()
    print(f"Plot saved as plot_{model_name.replace(' ', '_')}.png")
