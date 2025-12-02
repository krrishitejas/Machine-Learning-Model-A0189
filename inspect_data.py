import pandas as pd

try:
    df = pd.read_excel('2010 to 2019.xlsx')
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())
    print("Shape:", df.shape)
    print("Data Types:\n", df.dtypes)
except Exception as e:
    print(f"Error reading file: {e}")
