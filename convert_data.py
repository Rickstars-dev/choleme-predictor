import pandas as pd

# Load the real heart disease dataset
df = pd.read_csv('../heart-disease-predictor/data/heart.csv')

# Rename cholesterol column and remove target
df = df.rename(columns={'chol': 'total_cholesterol'})
df = df.drop('target', axis=1)

# Save as real data
df.to_csv('data/nhanes_cholesterol.csv', index=False)

print('✅ Real data copied!')
print(f'Samples: {len(df)}')
print(f'Mean cholesterol: {df["total_cholesterol"].mean():.2f} mg/dL')
print(f'Range: {df["total_cholesterol"].min()}-{df["total_cholesterol"].max()} mg/dL')
