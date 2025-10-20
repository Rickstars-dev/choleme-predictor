"""
Train cholesterol prediction model - Simplified for faster training
Using best practices without extensive hyperparameter tuning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CHOLESTEROL LEVEL PREDICTION - FAST TRAINING")
print("Dataset: Framingham Heart Study (4,240 samples)")
print("="*60)

# Load data
print("\nLoading dataset...")
df = pd.read_csv('data/nhanes_cholesterol.csv')
print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} features")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Separate features and target
X = df.drop('total_cholesterol', axis=1)
y = df['total_cholesterol']

print(f"\n📊 Target Statistics:")
print(f"   Mean: {y.mean():.2f} mg/dL")
print(f"   Median: {y.median():.2f} mg/dL")
print(f"   Range: {y.min():.0f} - {y.max():.0f} mg/dL")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# Scale features
print("\n🔄 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Train multiple models (using good default parameters)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

results = {}
best_model = None
best_score = -float('inf')
best_name = None

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'cv_r2': cv_mean
    }
    
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f} mg/dL")
    print(f"   RMSE: {rmse:.2f} mg/dL")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   CV R² (5-fold): {cv_mean:.4f}")
    
    # Track best model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Display comparison
print("\n📊 MODEL COMPARISON:")
print(f"{'Model':<25} {'R²':<10} {'MAE':<12} {'RMSE':<12} {'CV R²':<10}")
print("-" * 70)
for name, metrics in results.items():
    marker = "⭐" if name == best_name else "  "
    print(f"{marker} {name:<23} {metrics['r2']:<10.4f} {metrics['mae']:<12.2f} {metrics['rmse']:<12.2f} {metrics['cv_r2']:<10.4f}")

print(f"\n🏆 Best Model: {best_name}")
print(f"   Test R² Score: {best_score:.4f}")
print(f"   This means the model explains {best_score*100:.2f}% of cholesterol variation")

# Save the best model
print("\n💾 Saving model...")
os.makedirs('model', exist_ok=True)

joblib.dump(best_model, 'model/cholesterol_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')

# Save metadata
metadata = {
    'model_name': best_name,
    'r2_score': best_score,
    'mae': results[best_name]['mae'],
    'rmse': results[best_name]['rmse'],
    'mape': results[best_name]['mape'],
    'cv_r2': results[best_name]['cv_r2'],
    'target': 'total_cholesterol',
    'unit': 'mg/dL',
    'n_features': len(X.columns),
    'n_samples': len(df),
    'dataset': 'Framingham Heart Study'
}
joblib.dump(metadata, 'model/metadata.pkl')

print(f"✅ Model saved to: model/cholesterol_model.pkl")
print(f"✅ Scaler saved to: model/scaler.pkl")
print(f"✅ Metadata saved to: model/metadata.pkl")

print("\n" + "="*60)
print("🎉 SUCCESS! Your model is ready!")
print("="*60)
print("\nNext steps:")
print("1. Run the Flask app: python app.py")
print("2. Open browser: http://127.0.0.1:5000")
print("3. Test your cholesterol predictions!")
print("="*60)
