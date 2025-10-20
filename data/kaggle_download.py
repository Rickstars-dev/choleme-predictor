"""
Download cardiovascular dataset with ~5,000 samples using Kaggle Python API
"""

import os
import pandas as pd
import sys

# Configure Kaggle
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("✅ Kaggle API authenticated!\n")
except Exception as e:
    print(f"❌ Error authenticating: {e}")
    print("\nTrying manual authentication...")
    import kaggle
    print("✅ Kaggle module loaded!\n")

def download_framingham():
    """Download Framingham Heart Study - 4,240 samples"""
    print("="*60)
    print("DOWNLOADING: Framingham Heart Study")
    print("="*60)
    print("Samples: ~4,240 real patients")
    print("Features: Age, Sex, Cholesterol, BP, Smoking, etc.")
    print("="*60 + "\n")
    
    try:
        # Create temp directory
        os.makedirs('data/temp', exist_ok=True)
        
        print("🔄 Downloading...")
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(
            'aasheesh200/framingham-heart-study-dataset',
            path='data/temp',
            unzip=True
        )
        
        print("✅ Download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_heart_failure():
    """Download Heart Failure dataset - 918 samples (smaller but good quality)"""
    print("="*60)
    print("DOWNLOADING: Heart Failure Clinical Records")
    print("="*60)
    print("Samples: 299")
    print("="*60 + "\n")
    
    try:
        os.makedirs('data/temp', exist_ok=True)
        
        print("🔄 Downloading...")
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(
            'fedesoriano/heart-failure-prediction',
            path='data/temp',
            unzip=True
        )
        
        print("✅ Download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_combined_heart_disease():
    """Download combined heart disease dataset - 1,190 samples"""
    print("="*60)
    print("DOWNLOADING: Heart Disease Dataset (Combined)")
    print("="*60)
    print("Samples: ~1,190")
    print("="*60 + "\n")
    
    try:
        os.makedirs('data/temp', exist_ok=True)
        
        print("🔄 Downloading...")
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(
            'johnsmith88/heart-disease-dataset',
            path='data/temp',
            unzip=True
        )
        
        print("✅ Download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def process_data():
    """Process downloaded data"""
    print("\n" + "="*60)
    print("PROCESSING DATA")
    print("="*60 + "\n")
    
    import glob
    csv_files = glob.glob('data/temp/*.csv')
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    print(f"✅ Found {len(csv_files)} file(s)\n")
    
    all_data = []
    
    for csv_file in csv_files:
        print(f"📊 Loading: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        print(f"   Samples: {len(df):,}")
        print(f"   Columns: {list(df.columns)}\n")
        all_data.append(df)
    
    # Use the largest dataset
    df = max(all_data, key=len)
    
    print("="*60)
    print(f"📈 Selected Dataset: {len(df):,} samples")
    print("="*60)
    
    # Check for cholesterol column
    print("\nLooking for cholesterol-related columns...")
    
    chol_cols = [col for col in df.columns if any(term in col.lower() 
                 for term in ['chol', 'ldl', 'hdl', 'triglyc'])]
    
    if chol_cols:
        print(f"✅ Found: {chol_cols}")
        
        # Use 'chol' or first cholesterol column as target
        target_col = next((col for col in chol_cols if 'chol' in col.lower() and 'hdl' not in col.lower() and 'ldl' not in col.lower()), chol_cols[0])
        
        if target_col != 'total_cholesterol':
            df = df.rename(columns={target_col: 'total_cholesterol'})
            print(f"✅ Renamed '{target_col}' → 'total_cholesterol'")
        
        # Remove target column if exists
        if 'target' in df.columns:
            df = df.drop('target', axis=1)
            print("✅ Removed 'target' column")
        
        # Save
        output_file = 'data/nhanes_cholesterol.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Saved to: {output_file}")
        print(f"✅ Samples: {len(df):,}")
        print(f"✅ Features: {len(df.columns)}")
        
        # Show cholesterol stats
        if 'total_cholesterol' in df.columns:
            print(f"\n📊 Cholesterol Statistics:")
            print(f"   Mean: {df['total_cholesterol'].mean():.2f} mg/dL")
            print(f"   Median: {df['total_cholesterol'].median():.2f} mg/dL")
            print(f"   Range: {df['total_cholesterol'].min():.0f} - {df['total_cholesterol'].max():.0f} mg/dL")
        
        return True
    else:
        print("⚠️ No cholesterol column found")
        print(f"Available columns: {list(df.columns)}")
        
        # Save for inspection
        df.to_csv('data/downloaded_dataset.csv', index=False)
        print(f"\n💾 Saved to: data/downloaded_dataset.csv for inspection")
        return False

if __name__ == "__main__":
    print("="*60)
    print("KAGGLE DOWNLOADER - TARGET: ~5,000 SAMPLES")
    print("="*60 + "\n")
    
    # Try datasets in order of size
    datasets = [
        ("Framingham (~4,240 samples)", download_framingham),
        ("Combined Heart Disease (~1,190 samples)", download_combined_heart_disease),
        ("Heart Failure (299 samples)", download_heart_failure)
    ]
    
    success = False
    for name, download_func in datasets:
        print(f"🔄 Attempting: {name}\n")
        if download_func():
            success = True
            break
        print("\n⚠️ Failed, trying next option...\n")
    
    if success:
        process_data()
        print("\n✅ Ready to train! Run: python train_model.py")
    else:
        print("\n❌ All download attempts failed")
        print("💡 You can manually download from https://www.kaggle.com/datasets")
