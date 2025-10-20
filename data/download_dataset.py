"""
Download NHANES cholesterol dataset from Kaggle

IMPORTANT: You need a Kaggle account and API key to download datasets
Setup instructions: https://github.com/Kaggle/kaggle-api#api-credentials
"""
import os
import sys

def check_kaggle_setup():
    """Check if Kaggle is properly configured"""
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle API not configured!")
        print("\n📋 Setup Instructions:")
        print("1. Go to https://www.kaggle.com/")
        print("2. Sign in (or create account)")
        print("3. Go to Account Settings")
        print("4. Scroll to 'API' section")
        print("5. Click 'Create New API Token'")
        print("6. Save the downloaded 'kaggle.json' to: ~/.kaggle/")
        print("   Windows: C:\\Users\\YourName\\.kaggle\\kaggle.json")
        print("7. Run this script again")
        return False
    return True

def download_nhanes_dataset():
    """
    Download NHANES dataset from Kaggle
    """
    print("="*60)
    print("DOWNLOADING NHANES CHOLESTEROL DATASET")
    print("Source: Kaggle - CDC NHANES Data")
    print("="*60)
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        return
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        print("\n✅ Kaggle API authenticated!")
        print("\n🔄 Searching for NHANES datasets...")
        
        # Option 1: Try downloading a specific NHANES dataset
        # You may need to replace this with an actual dataset identifier
        dataset_options = [
            'cdc/national-health-and-nutrition-examination-survey',
            'cdc/nhanes',
            # Add more options as fallbacks
        ]
        
        success = False
        for dataset_id in dataset_options:
            try:
                print(f"\n🔄 Trying to download: {dataset_id}")
                api.dataset_download_files(dataset_id, path='.', unzip=True)
                print(f"✅ Downloaded: {dataset_id}")
                success = True
                break
            except Exception as e:
                print(f"   ⚠️  Not available: {dataset_id}")
                continue
        
        if not success:
            print("\n⚠️  No NHANES dataset found via API")
            print("\n📥 MANUAL DOWNLOAD OPTION:")
            print("1. Visit: https://www.kaggle.com/datasets")
            print("2. Search for: 'NHANES cholesterol' or 'CDC health survey'")
            print("3. Download the CSV file")
            print("4. Save it as: data/nhanes_cholesterol.csv")
            print("\n💡 Alternative: Create synthetic dataset")
            create_synthetic_dataset()
        else:
            # Rename downloaded file
            rename_dataset_files()
            
    except ImportError:
        print("\n❌ Kaggle package not installed!")
        print("Run: pip install kaggle")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Creating synthetic dataset as fallback...")
        create_synthetic_dataset()

def rename_dataset_files():
    """
    Rename downloaded files to expected name
    """
    print("\n🔄 Looking for downloaded files...")
    
    # Look for CSV files
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            print(f"   Found: {file}")
            if 'cholesterol' in file.lower() or 'lab' in file.lower():
                os.rename(file, 'nhanes_cholesterol.csv')
                print(f"✅ Renamed to: nhanes_cholesterol.csv")
                return
    
    print("⚠️  No suitable file found")

def create_synthetic_dataset():
    """
    Create a synthetic NHANES-like dataset for development
    This is a fallback if Kaggle download fails
    """
    print("\n🔄 Creating synthetic NHANES-like dataset...")
    
    try:
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic health data
        data = {
            'age': np.random.randint(20, 80, n_samples),
            'gender': np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
            'bmi': np.random.normal(28, 5, n_samples).clip(18, 50),
            'systolic_bp': np.random.normal(130, 15, n_samples).clip(90, 200),
            'diastolic_bp': np.random.normal(85, 10, n_samples).clip(60, 130),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'physical_activity': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),  # 0=Low, 1=Moderate, 2=High
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'alcohol_drinks_per_week': np.random.randint(0, 15, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate cholesterol based on other features (with some randomness)
        base_cholesterol = 160
        df['total_cholesterol'] = (
            base_cholesterol +
            (df['age'] - 20) * 0.8 +  # Age effect
            df['gender'] * 10 +  # Gender effect (males slightly higher)
            (df['bmi'] - 25) * 2 +  # BMI effect
            (df['systolic_bp'] - 120) * 0.3 +  # Blood pressure effect
            df['smoking'] * 15 +  # Smoking effect
            -df['physical_activity'] * 10 +  # Physical activity effect (negative)
            df['diabetes'] * 20 +  # Diabetes effect
            df['alcohol_drinks_per_week'] * 0.5 +  # Alcohol effect
            np.random.normal(0, 15, n_samples)  # Random variation
        ).clip(120, 350)  # Realistic range
        
        # Save to CSV
        df.to_csv('nhanes_cholesterol.csv', index=False)
        
        print("✅ Synthetic dataset created: nhanes_cholesterol.csv")
        print(f"   Samples: {n_samples}")
        print(f"   Features: {list(df.columns)}")
        print(f"\n📊 Cholesterol Statistics:")
        print(f"   Mean: {df['total_cholesterol'].mean():.2f} mg/dL")
        print(f"   Median: {df['total_cholesterol'].median():.2f} mg/dL")
        print(f"   Range: {df['total_cholesterol'].min():.0f}-{df['total_cholesterol'].max():.0f} mg/dL")
        
        print("\n⚠️  NOTE: This is synthetic data for development/testing")
        print("For production, use real NHANES data from CDC/Kaggle")
        
    except ImportError:
        print("❌ pandas/numpy not installed. Run: pip install pandas numpy")
    except Exception as e:
        print(f"❌ Error creating synthetic dataset: {e}")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("NHANES CHOLESTEROL DATASET DOWNLOADER")
    print("="*60)
    
    # Try Kaggle download first
    download_nhanes_dataset()
    
    # Check if we have a dataset
    if os.path.exists('nhanes_cholesterol.csv'):
        print("\n✅ SUCCESS! Dataset ready for training")
        print("Next step: Run 'python train_model.py'")
    else:
        print("\n❌ No dataset available")
        print("Please download manually or configure Kaggle API")

if __name__ == "__main__":
    main()
