import pandas as pd
import numpy as np

file_path = '2010 to 2019.xlsx'
df = pd.read_excel(file_path)

# Preprocessing
if 'WATER LEVEL (M)' in df.columns:
    df['WATER LEVEL (M)'] = pd.to_numeric(df['WATER LEVEL (M)'], errors='coerce')

df = df.dropna(subset=['Seepage (l/day)'])
df = df.fillna(df.mean())

print("Correlation with Seepage (l/day):")
print(df.corr()['Seepage (l/day)'].sort_values(ascending=False))

print("\nTarget Description:")
print(df['Seepage (l/day)'].describe())
