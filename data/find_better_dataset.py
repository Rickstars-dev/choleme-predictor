"""
Search and download better cholesterol datasets from Kaggle
Looking for datasets with: diet, medication, family history, LDL/HDL breakdown
"""

import subprocess
import pandas as pd
import os
import sys

def search_cholesterol_datasets():
    """Search for specialized cholesterol datasets"""
    print("="*60)
    print("SEARCHING KAGGLE FOR BETTER CHOLESTEROL DATASETS")
    print("="*60)
    
    print("\n🔍 Searching for datasets with cholesterol-specific features...\n")
    
    search_terms = [
        "cholesterol ldl hdl",
        "lipid profile prediction",
        "cardiovascular risk cholesterol",
        "nhanes cholesterol complete",
        "diet cholesterol prediction"
    ]
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    all_datasets = []
    
    for term in search_terms:
        print(f"\n📊 Searching: '{term}'")
        try:
            datasets = api.dataset_list(search=term, sort_by='votes')
            for ds in datasets[:5]:  # Top 5 for each search
                dataset_info = {
                    'ref': ds.ref,
                    'title': ds.title,
                    'size': ds.totalBytes,
                    'votes': ds.voteCount,
                    'url': f"https://www.kaggle.com/datasets/{ds.ref}"
                }
                if dataset_info not in all_datasets:
                    all_datasets.append(dataset_info)
                    print(f"   ✓ {ds.title}")
                    print(f"     Ref: {ds.ref}")
                    print(f"     Size: {ds.totalBytes / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    
    return all_datasets

def download_dataset(dataset_ref):
    """Download a specific dataset"""
    print(f"\n🔄 Downloading: {dataset_ref}")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        os.makedirs('data/temp', exist_ok=True)
        
        api.dataset_download_files(
            dataset_ref,
            path='data/temp',
            unzip=True
        )
        
        print("✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def try_promising_datasets():
    """Try downloading known good cholesterol datasets"""
    
    # List of promising datasets for cholesterol prediction
    promising = [
        {
            'ref': 'rashikrahmanpritom/heart-attack-analysis-prediction-dataset',
            'name': 'Heart Attack Analysis',
            'expected_features': 'cholesterol, age, sex, thalach, oldpeak'
        },
        {
            'ref': 'sulianova/cardiovascular-disease-dataset',
            'name': 'Cardiovascular Disease',
            'expected_features': 'cholesterol, glucose, weight, height, smoke'
        },
        {
            'ref': 'johnsmith88/heart-disease-dataset',
            'name': 'Combined Heart Disease',
            'expected_features': 'cholesterol levels from multiple sources'
        }
    ]
    
    print("\n" + "="*60)
    print("TRYING PROMISING DATASETS")
    print("="*60)
    
    for ds in promising:
        print(f"\n📦 Dataset: {ds['name']}")
        print(f"   Reference: {ds['ref']}")
        print(f"   Expected: {ds['expected_features']}")
        print(f"\n   Attempting download...")
        
        if download_dataset(ds['ref']):
            # Check what we got
            import glob
            csv_files = glob.glob('data/temp/*.csv')
            
            if csv_files:
                print(f"\n   ✅ Found {len(csv_files)} CSV file(s)")
                
                for csv_file in csv_files:
                    print(f"\n   📄 File: {os.path.basename(csv_file)}")
                    df = pd.read_csv(csv_file)
                    print(f"      Samples: {len(df):,}")
                    print(f"      Columns: {list(df.columns)}")
                    
                    # Check for cholesterol column
                    chol_cols = [col for col in df.columns if 'chol' in col.lower()]
                    if chol_cols:
                        print(f"      ✅ Cholesterol columns: {chol_cols}")
                        print(f"\n      ⭐ This looks good!")
                        return csv_file, ds['ref']
                    else:
                        print(f"      ⚠️ No cholesterol column found")
            
            # Clean up for next attempt
            for csv_file in csv_files:
                os.remove(csv_file)
        
        print("\n   " + "-"*50)
    
    return None, None

def analyze_dataset(csv_path):
    """Analyze dataset quality for cholesterol prediction"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 Dataset: {os.path.basename(csv_path)}")
    print(f"   Samples: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    
    print(f"\n📋 All Columns:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        print(f"   {i:2d}. {col:<30} ({dtype}, {unique} unique, {missing} missing)")
    
    # Find cholesterol column
    chol_cols = [col for col in df.columns if 'chol' in col.lower()]
    
    if chol_cols:
        print(f"\n💚 Cholesterol columns found: {chol_cols}")
        
        for col in chol_cols:
            print(f"\n   📈 {col}:")
            print(f"      Mean: {df[col].mean():.2f}")
            print(f"      Median: {df[col].median():.2f}")
            print(f"      Range: {df[col].min():.0f} - {df[col].max():.0f}")
            print(f"      Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        
        return True
    else:
        print("\n❌ No cholesterol column found!")
        return False

if __name__ == "__main__":
    print("="*60)
    print("KAGGLE CHOLESTEROL DATASET FINDER")
    print("="*60)
    
    # Try promising datasets first
    csv_file, dataset_ref = try_promising_datasets()
    
    if csv_file:
        print("\n" + "="*60)
        print("✅ FOUND SUITABLE DATASET!")
        print("="*60)
        
        # Analyze it
        if analyze_dataset(csv_file):
            print("\n🎯 This dataset can be used for cholesterol prediction!")
            print(f"📍 Location: {csv_file}")
            print(f"📦 Source: {dataset_ref}")
            
            # Copy to main data folder
            final_path = 'data/cholesterol_dataset_new.csv'
            df = pd.read_csv(csv_file)
            df.to_csv(final_path, index=False)
            print(f"\n✅ Copied to: {final_path}")
            print("\nNext: Run training with this new dataset!")
    else:
        print("\n⚠️ No suitable dataset found in quick search")
        print("\nSearching all datasets...")
        datasets = search_cholesterol_datasets()
        
        print("\n" + "="*60)
        print(f"FOUND {len(datasets)} DATASETS")
        print("="*60)
        
        print("\nTop 10 by votes:")
        for i, ds in enumerate(sorted(datasets, key=lambda x: x['votes'], reverse=True)[:10], 1):
            print(f"\n{i}. {ds['title']}")
            print(f"   Ref: {ds['ref']}")
            print(f"   Size: {ds['size']/(1024*1024):.2f} MB")
            print(f"   Votes: {ds['votes']}")
            print(f"   URL: {ds['url']}")
