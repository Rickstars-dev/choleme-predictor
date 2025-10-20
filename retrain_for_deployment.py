"""
Quick retrain script for deployment compatibility
Trains model without cross-validation to avoid issues
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

print("=" * 60)
print("RETRAINING MODEL FOR DEPLOYMENT")
print("=" * 60)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/nhanes_cholesterol.csv')
print(f"✅ Loaded {len(df)} samples")

# Prepare features and target
X = df.drop('total_cholesterol', axis=1)
y = df['total_cholesterol']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
print("\n🔄 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest (best model)
print("\n🌲 Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=1  # Use single job to avoid issues
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   R² Score: {r2:.4f}")
print(f"   MAE: {mae:.2f} mg/dL")

# Save model files
print("\n💾 Saving model files...")
os.makedirs('model', exist_ok=True)

joblib.dump(model, 'model/cholesterol_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')

metadata = {
    'model_name': 'Random Forest Regressor',
    'target': 'Cholesterol Level',
    'unit': 'mg/dL',
    'r2_score': r2,
    'mae': mae,
    'samples': len(df)
}
joblib.dump(metadata, 'model/metadata.pkl')

print("✅ All model files saved successfully!")
print("\n" + "=" * 60)
print("MODEL READY FOR DEPLOYMENT")
print("=" * 60)
