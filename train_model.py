"""
Train the cholesterol level prediction model using NHANES data
This is a REGRESSION model (predicts continuous values, not categories)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    Load the NHANES cholesterol dataset
    """
    print("Loading NHANES dataset...")
    
    try:
        # Load the cleaned NHANES dataset
        df = pd.read_csv('data/nhanes_cholesterol.csv')
        print(f"✅ Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print("\n❌ Error: Dataset not found!")
        print("\nPlease run: python data/download_dataset.py")
        print("This will download the NHANES dataset from Kaggle")
        raise

def preprocess_data(df, target_column='total_cholesterol'):
    """
    Preprocess the dataset for regression
    """
    print("\nPreprocessing data...")
    
    # Display dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Handle missing values
    print("\nHandling missing values...")
    df = df.fillna(df.mean(numeric_only=True))
    
    # Separate features and target
    if target_column not in df.columns:
        print(f"\n⚠️  Target column '{target_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        # Try common cholesterol column names
        possible_names = ['cholesterol', 'chol', 'total_chol', 'LBXTC', 'serum_cholesterol']
        for name in possible_names:
            if name in df.columns:
                target_column = name
                print(f"✅ Using '{target_column}' as target")
                break
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"\nTarget variable: {target_column}")
    print(f"Target statistics:")
    print(f"  Mean: {y.mean():.2f} mg/dL")
    print(f"  Median: {y.median():.2f} mg/dL")
    print(f"  Min: {y.min():.2f} mg/dL")
    print(f"  Max: {y.max():.2f} mg/dL")
    print(f"  Std: {y.std():.2f} mg/dL")
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Scale the features (important for regression!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)

def train_models(X_train, y_train):
    """
    Train multiple regression models with hyperparameter tuning
    """
    print("\n" + "="*60)
    print("TRAINING & TUNING REGRESSION MODELS")
    print("Using GridSearchCV for optimal hyperparameters")
    print("="*60)
    
    trained_models = {}
    
    # 1. Linear Regression (no hyperparameters to tune)
    print(f"\n🔄 Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    trained_models['Linear Regression'] = lr_model
    print(f"✅ Linear Regression trained!")
    
    # 2. Ridge Regression (tune alpha)
    print(f"\n🔄 Tuning Ridge Regression...")
    print("   Testing alpha values: [0.01, 0.1, 1.0, 10, 100]")
    ridge_params = {
        'alpha': [0.01, 0.1, 1.0, 10, 50, 100]
    }
    ridge_grid = GridSearchCV(
        Ridge(random_state=42),
        ridge_params,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    ridge_grid.fit(X_train, y_train)
    trained_models['Ridge Regression'] = ridge_grid.best_estimator_
    print(f"✅ Best alpha: {ridge_grid.best_params_['alpha']}")
    print(f"   Best CV R²: {ridge_grid.best_score_:.4f}")
    
    # 3. Random Forest (tune multiple parameters)
    print(f"\n🔄 Tuning Random Forest...")
    print("   Testing multiple combinations (this may take a minute)...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train, y_train)
    trained_models['Random Forest'] = rf_grid.best_estimator_
    print(f"✅ Best params:")
    for param, value in rf_grid.best_params_.items():
        print(f"   {param}: {value}")
    print(f"   Best CV R²: {rf_grid.best_score_:.4f}")
    
    # 4. Gradient Boosting (tune multiple parameters)
    print(f"\n🔄 Tuning Gradient Boosting...")
    print("   Testing multiple combinations (this may take a minute)...")
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10]
    }
    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    gb_grid.fit(X_train, y_train)
    trained_models['Gradient Boosting'] = gb_grid.best_estimator_
    print(f"✅ Best params:")
    for param, value in gb_grid.best_params_.items():
        print(f"   {param}: {value}")
    print(f"   Best CV R²: {gb_grid.best_score_:.4f}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and compare performance
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    
    for name, model in models.items():
        print(f"\n📊 {name}:")
        print("-" * 40)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[name] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        # Display metrics
        print(f"  R² Score:  {r2:.4f} ({r2*100:.2f}% variance explained)")
        print(f"  MAE:       {mae:.2f} mg/dL (average error)")
        print(f"  RMSE:      {rmse:.2f} mg/dL (weighted error)")
        print(f"  MAPE:      {mape:.2f}% (percentage error)")
        
        # Interpretation
        if r2 >= 0.80:
            print(f"  Rating:    ⭐⭐⭐⭐⭐ Excellent!")
        elif r2 >= 0.70:
            print(f"  Rating:    ⭐⭐⭐⭐ Very Good!")
        elif r2 >= 0.60:
            print(f"  Rating:    ⭐⭐⭐ Good")
        else:
            print(f"  Rating:    ⭐⭐ Moderate")
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {results[best_model_name]['r2']:.4f}")
    print(f"   MAE: {results[best_model_name]['mae']:.2f} mg/dL")
    print("="*60)
    
    return models[best_model_name], best_model_name, results

def save_model(model, scaler, feature_names, model_name):
    """
    Save the trained model, scaler, and metadata
    """
    print(f"\n💾 Saving {model_name}...")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, 'model/cholesterol_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(feature_names, 'model/feature_names.pkl')
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'features': feature_names,
        'target': 'total_cholesterol',
        'unit': 'mg/dL'
    }
    joblib.dump(metadata, 'model/metadata.pkl')
    
    print("✅ Model saved successfully!")
    print(f"   - model/cholesterol_model.pkl")
    print(f"   - model/scaler.pkl")
    print(f"   - model/feature_names.pkl")
    print(f"   - model/metadata.pkl")

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("CHOLESTEROL LEVEL PREDICTION - MODEL TRAINING")
    print("Dataset: NHANES (CDC Government Data)")
    print("Model Type: REGRESSION (predicts continuous values)")
    print("="*60)
    
    try:
        # Load data
        df = load_data()
        
        # Preprocess
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
        
        # Train multiple models
        trained_models = train_models(X_train, y_train)
        
        # Evaluate and select best model
        best_model, best_model_name, results = evaluate_models(trained_models, X_test, y_test)
        
        # Save the best model
        save_model(best_model, scaler, feature_names, best_model_name)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Best Model: {best_model_name}")
        print(f"R² Score: {results[best_model_name]['r2']:.4f}")
        print(f"MAE: {results[best_model_name]['mae']:.2f} mg/dL")
        print(f"\nThis means predictions will be off by ~{results[best_model_name]['mae']:.0f} mg/dL on average")
        print("\nYou can now run: python app.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
