"""
Improved cholesterol prediction with feature engineering
Creating new features to improve R² score
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
print("IMPROVED CHOLESTEROL PREDICTION - FEATURE ENGINEERING")
print("="*60)

# Load data
print("\nLoading dataset...")
df = pd.read_csv('data/nhanes_cholesterol.csv')
print(f"✅ Loaded {len(df):,} samples")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

print("\n🔧 Creating engineered features...")

# 1. Age groups (cholesterol increases with age)
df['age_squared'] = df['age'] ** 2
df['age_group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 100], labels=[0, 1, 2, 3])

# 2. BMI categories
df['bmi_squared'] = df['BMI'] ** 2
df['obesity'] = (df['BMI'] >= 30).astype(int)
df['overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)

# 3. Blood Pressure interactions
df['bp_product'] = df['sysBP'] * df['diaBP']
df['pulse_pressure'] = df['sysBP'] - df['diaBP']
df['mean_arterial_pressure'] = (df['sysBP'] + 2 * df['diaBP']) / 3

# 4. Cardiovascular risk factors
df['risk_score'] = (
    df['currentSmoker'] * 2 +
    df['diabetes'] * 3 +
    df['prevalentHyp'] * 2 +
    df['prevalentStroke'] * 3 +
    (df['age'] > 55).astype(int) * 1
)

# 5. Metabolic features
df['glucose_bmi_interaction'] = df['glucose'] * df['BMI']
df['glucose_age_interaction'] = df['glucose'] * df['age']

# 6. Smoking intensity
df['smoking_intensity'] = df['currentSmoker'] * df['cigsPerDay']

# 7. Heart rate categories
df['high_heart_rate'] = (df['heartRate'] > 85).astype(int)
df['low_heart_rate'] = (df['heartRate'] < 60).astype(int)

# 8. Multiple risk factors
df['multiple_risks'] = (
    df['currentSmoker'] +
    df['diabetes'] +
    df['prevalentHyp'] +
    (df['BMI'] >= 30).astype(int)
)

# 9. Age-gender interaction
df['age_male_interaction'] = df['age'] * df['male']

# 10. Education-lifestyle proxy
df['edu_smoking_interaction'] = df['education'] * (1 - df['currentSmoker'])

print(f"✅ Created {len(df.columns)} total features (was 16, now {len(df.columns)})")

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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("TRAINING ENHANCED MODELS")
print("="*60)

# Enhanced models with better parameters
models = {
    'Ridge (Enhanced)': Ridge(alpha=1.0),
    'Random Forest (Deep)': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting (Strong)': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.8,
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
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_r2': cv_mean
    }
    
    print(f"   R² Score: {r2:.4f} {'🚀' if r2 > 0.15 else ''}")
    print(f"   MAE: {mae:.2f} mg/dL")
    print(f"   CV R² (5-fold): {cv_mean:.4f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print("\n" + "="*60)
print("RESULTS WITH FEATURE ENGINEERING")
print("="*60)

print(f"\n{'Model':<30} {'R²':<10} {'MAE':<12} {'CV R²':<10}")
print("-" * 65)
for name, metrics in results.items():
    marker = "⭐" if name == best_name else "  "
    print(f"{marker} {name:<28} {metrics['r2']:<10.4f} {metrics['mae']:<12.2f} {metrics['cv_r2']:<10.4f}")

improvement = best_score - 0.1159  # Previous best
print(f"\n📈 Improvement: +{improvement:.4f} R² ({improvement*100:.2f}%)")
print(f"🏆 Best Model: {best_name}")
print(f"   R² Score: {best_score:.4f} ({best_score*100:.2f}%)")

# Save model
print("\n💾 Saving improved model...")
os.makedirs('model', exist_ok=True)

joblib.dump(best_model, 'model/cholesterol_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')

metadata = {
    'model_name': best_name,
    'r2_score': best_score,
    'mae': results[best_name]['mae'],
    'cv_r2': results[best_name]['cv_r2'],
    'target': 'total_cholesterol',
    'unit': 'mg/dL',
    'n_features': len(X.columns),
    'n_samples': len(df),
    'dataset': 'Framingham Heart Study (Enhanced)',
    'feature_engineering': True
}
joblib.dump(metadata, 'model/metadata.pkl')

print("✅ Model saved!")

# Feature importance for Random Forest
if 'Random Forest' in best_name:
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")

print("\n" + "="*60)
print("🎉 ENHANCED MODEL READY!")
print("="*60)
