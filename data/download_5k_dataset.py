"""
Download cardiovascular dataset with ~5,000 samples from Kaggle
"""

import subprocess
import os
import pandas as pd
import sys

def download_framingham_dataset():
    """
    Download Framingham Heart Study dataset
    Size: ~4,240 samples (close to 5,000)
    Real medical data with cholesterol measurements
    """
    print("="*60)
    print("DOWNLOADING: Framingham Heart Study Dataset")
    print("="*60)
    print("Samples: ~4,240 real patients")
    print("Source: Framingham Heart Study (Longitudinal study)")
    print("Features: Age, Sex, Cholesterol, BP, Smoking, Diabetes, BMI")
    print("="*60 + "\n")
    
    try:
        print("🔄 Downloading from Kaggle...")
        
        # Download Framingham dataset
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'aasheesh200/framingham-heart-study-dataset', 
             '-p', 'data/temp', '--unzip'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Download successful!")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_heart_disease_comprehensive():
    """
    Alternative: Comprehensive heart disease dataset
    Multiple sources combined: ~5,000+ samples
    """
    print("\n" + "="*60)
    print("ALTERNATIVE: Comprehensive Heart Disease Dataset")
    print("="*60)
    
    datasets_to_try = [
        'aasheesh200/framingham-heart-study-dataset',
        'fedesoriano/heart-failure-prediction',
        'johnsmith88/heart-disease-dataset',
        'rashikrahmanpritom/heart-attack-analysis-prediction-dataset'
    ]
    
    for dataset in datasets_to_try:
        print(f"\n🔄 Trying: {dataset}")
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset, 
                 '-p', 'data/temp', '--unzip'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully downloaded: {dataset}")
                return dataset
            else:
                print(f"⚠️ Not available: {dataset}")
        except Exception as e:
            print(f"⚠️ Skipped: {e}")
            continue
    
    return None

def process_downloaded_data():
    """Process the downloaded data to create cholesterol prediction dataset"""
    print("\n" + "="*60)
    print("PROCESSING DOWNLOADED DATA")
    print("="*60 + "\n")
    
    # Look for CSV files in temp directory
    import glob
    
    csv_files = glob.glob('data/temp/*.csv')
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    print(f"✅ Found {len(csv_files)} CSV file(s)")
    
    # Load the largest CSV file
    largest_file = max(csv_files, key=os.path.getsize)
    print(f"📊 Loading: {os.path.basename(largest_file)}")
    
    df = pd.read_csv(largest_file)
    
    print(f"\n📈 Dataset Info:")
    print(f"   Samples: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    # Check if cholesterol column exists
    chol_cols = [col for col in df.columns if 'chol' in col.lower() or 'ldl' in col.lower() or 'hdl' in col.lower()]
    
    if chol_cols:
        print(f"\n✅ Found cholesterol columns: {chol_cols}")
        
        # Save to main data folder
        output_file = 'data/nhanes_cholesterol.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        print(f"✅ Ready for training with {len(df):,} samples!")
        return True
    else:
        print("\n⚠️ No cholesterol column found. Columns available:")
        print(df.columns.tolist())
        
        # Save anyway for manual inspection
        output_file = 'data/downloaded_dataset.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file} for inspection")
        return False

def search_datasets():
    """Search for datasets with specific size"""
    print("🔍 Searching for cardiovascular datasets (~5000 samples)...\n")
    
    result = subprocess.run(
        ['kaggle', 'datasets', 'list', '-s', 'heart disease', '--csv'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Available datasets:")
        print(result.stdout[:2000])  # First 2000 chars
    else:
        print(f"Error: {result.stderr}")

if __name__ == "__main__":
    print("="*60)
    print("KAGGLE DATASET DOWNLOADER - TARGET SIZE: 5,000 SAMPLES")
    print("="*60 + "\n")
    
    # Create temp directory
    os.makedirs('data/temp', exist_ok=True)
    
    # Try to download Framingham (best match for 5,000 samples)
    success = download_framingham_dataset()
    
    if not success:
        print("\n⚠️ Framingham download failed. Trying alternatives...")
        dataset_name = download_heart_disease_comprehensive()
        
        if not dataset_name:
            print("\n❌ Could not download any dataset")
            print("\n💡 Trying search instead...")
            search_datasets()
            sys.exit(1)
    
    # Process the downloaded data
    print("\n" + "="*60)
    if process_downloaded_data():
        print("\n🎉 SUCCESS! Dataset ready for training!")
        print("\nNext step: Run 'python train_model.py' to train with new data")
    else:
        print("\n⚠️ Manual processing may be needed")
