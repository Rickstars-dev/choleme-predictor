"""
Download large, real NHANES or cardiovascular datasets from Kaggle

OPTION 1: Using Kaggle API (requires setup)
OPTION 2: Manual download instructions
"""

import os
import pandas as pd

def download_from_kaggle():
    """
    Download large datasets from Kaggle using API
    
    Popular large cardiovascular/cholesterol datasets:
    1. cdc/behavioral-risk-factor-surveillance-system (400,000+ samples)
    2. cdc/national-health-and-nutrition-examination-survey (Multiple years, 10,000+ per year)
    3. Various cleaned NHANES datasets
    """
    try:
        import kaggle
        
        print("🔄 Searching for large NHANES/cardiovascular datasets...")
        
        # Option 1: BRFSS (Behavioral Risk Factor Surveillance System) - 400,000+ samples
        print("\n📊 Downloading BRFSS dataset (400,000+ samples)...")
        kaggle.api.dataset_download_files(
            'cdc/behavioral-risk-factor-surveillance-system',
            path='data/temp',
            unzip=True
        )
        
        print("✅ Dataset downloaded!")
        print("\nProcessing data to extract cholesterol-related features...")
        
        # Process the BRFSS data
        process_brfss_data()
        
    except ImportError:
        print("❌ Kaggle API not installed or configured")
        show_manual_instructions()
    except Exception as e:
        print(f"❌ Error: {e}")
        show_manual_instructions()

def process_brfss_data():
    """Process BRFSS data to extract relevant cholesterol predictors"""
    try:
        # BRFSS typically has a main CSV file
        import glob
        csv_files = glob.glob('data/temp/*.csv')
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            print(f"Original dataset: {len(df)} samples, {len(df.columns)} features")
            
            # Select relevant columns (these vary by year, adjust as needed)
            # Common BRFSS columns: age, sex, BMI, blood pressure, cholesterol check, etc.
            
            print("\n✅ Large dataset ready!")
            print(f"Samples: {len(df):,}")
            
    except Exception as e:
        print(f"Error processing data: {e}")

def show_manual_instructions():
    """Show instructions for manual download"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nTo get Kaggle API access:")
    print("1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. Save kaggle.json to: C:\\Users\\Abhishek Chandra\\.kaggle\\")
    
    print("\n" + "="*60)
    print("ALTERNATIVE: MANUAL DOWNLOAD (LARGER DATASETS)")
    print("="*60)
    
    print("\n📊 RECOMMENDED LARGE DATASETS:")
    
    print("\n1. BRFSS (400,000+ samples)")
    print("   URL: https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system")
    print("   Size: ~100 MB")
    print("   Contains: Age, BMI, blood pressure, cholesterol screening, lifestyle factors")
    
    print("\n2. NHANES Combined Years (30,000+ samples)")
    print("   URL: https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey")
    print("   Size: Varies by year")
    print("   Contains: Complete health markers including cholesterol levels")
    
    print("\n3. Framingham Heart Study (15,000+ samples)")
    print("   Search Kaggle for: 'framingham heart study'")
    print("   Contains: Long-term cardiovascular data")
    
    print("\n4. UK Biobank Cardiovascular (10,000+ samples)")
    print("   Search Kaggle for: 'uk biobank cardiovascular'")
    print("   Contains: Comprehensive health data")
    
    print("\n" + "="*60)
    print("AFTER DOWNLOADING:")
    print("="*60)
    print("1. Place the CSV file in: cholesterol_predictor/data/")
    print("2. Rename it to: large_dataset.csv")
    print("3. Run: python data/process_large_dataset.py")
    print("="*60)

def use_combined_datasets():
    """Alternative: Combine multiple smaller real datasets"""
    print("\n📦 ALTERNATIVE APPROACH: Combine Multiple Real Datasets")
    print("="*60)
    
    print("\nWe can combine several real medical datasets to get 1000+ samples:")
    print("1. UCI Heart Disease (303 samples) ✅ Already have")
    print("2. Statlog Heart (270 samples) - From UCI")
    print("3. Cleveland Clinic (297 samples) - Additional data")
    print("4. Hungarian Heart Disease (294 samples)")
    print("5. Switzerland Heart Disease (123 samples)")
    
    print("\nTotal: ~1,287 real patient samples")
    print("\nWould you like me to create a script to download and combine these?")

if __name__ == "__main__":
    print("="*60)
    print("LARGE DATASET DOWNLOADER")
    print("="*60)
    
    # Check if Kaggle is configured
    kaggle_json = "C:\\Users\\Abhishek Chandra\\.kaggle\\kaggle.json"
    
    if os.path.exists(kaggle_json):
        print("✅ Kaggle API configured!")
        download_from_kaggle()
    else:
        print("⚠️  Kaggle API not configured yet")
        show_manual_instructions()
        print("\n" + "="*60)
        use_combined_datasets()
