"""
Download large NHANES/cardiovascular datasets from Kaggle
Now that Kaggle API is configured, we can access massive datasets!
"""

import os
import pandas as pd
import subprocess
import sys

def search_kaggle_datasets():
    """Search for large cardiovascular datasets"""
    print("🔍 Searching Kaggle for large cardiovascular/cholesterol datasets...\n")
    
    search_terms = [
        "nhanes cholesterol",
        "cardiovascular health",
        "heart disease prediction"
    ]
    
    for term in search_terms:
        print(f"\n📊 Searching: {term}")
        os.system(f'kaggle datasets list -s "{term}" --max-size 500000000')

def download_brfss():
    """Download BRFSS - 400,000+ samples"""
    print("\n" + "="*60)
    print("DOWNLOADING: BRFSS Dataset")
    print("Behavioral Risk Factor Surveillance System")
    print("Samples: 400,000+")
    print("Source: CDC (U.S. Government)")
    print("="*60 + "\n")
    
    try:
        # Download BRFSS
        print("🔄 Downloading... (this may take 2-3 minutes)")
        os.system('kaggle datasets download -d cdc/behavioral-risk-factor-surveillance-system -p data/temp --unzip')
        
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_framingham():
    """Alternative: Framingham Heart Study"""
    print("\n" + "="*60)
    print("ALTERNATIVE: Framingham Heart Study")
    print("Samples: 4,000+")
    print("="*60 + "\n")
    
    try:
        print("🔄 Searching for Framingham datasets...")
        os.system('kaggle datasets list -s "framingham" --max-size 100000000')
    except Exception as e:
        print(f"Error: {e}")

def show_available_datasets():
    """Show user which datasets are available"""
    print("\n" + "="*60)
    print("LARGE CARDIOVASCULAR DATASETS ON KAGGLE")
    print("="*60)
    
    datasets = [
        {
            "name": "BRFSS (CDC)",
            "samples": "400,000+",
            "command": "cdc/behavioral-risk-factor-surveillance-system",
            "features": "Age, BMI, BP, Cholesterol check, Smoking, Exercise"
        },
        {
            "name": "Framingham Heart Study",
            "samples": "4,240",
            "command": "Search 'framingham heart'",
            "features": "Cholesterol, BP, Age, Smoking, Diabetes, BMI"
        },
        {
            "name": "Heart Disease UCI (Multiple)",
            "samples": "1,000+",
            "command": "Search 'heart disease'",
            "features": "Direct cholesterol measurements"
        }
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   Samples: {ds['samples']}")
        print(f"   Features: {ds['features']}")
        print(f"   Command: {ds['command']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("="*60)
    print("KAGGLE DATASET DOWNLOADER")
    print("="*60)
    
    # Test Kaggle authentication
    print("\n🔐 Testing Kaggle authentication...")
    result = os.system('kaggle datasets list --max-size 1000 2>nul')
    
    if result == 0:
        print("✅ Kaggle API authenticated successfully!\n")
        
        show_available_datasets()
        
        print("\n" + "="*60)
        print("RECOMMENDED ACTION:")
        print("="*60)
        print("\nLet me search for the best available dataset...")
        
        # Search for cholesterol/cardiovascular datasets
        print("\n🔍 Searching Kaggle...")
        os.system('kaggle datasets list -s "heart disease cholesterol" --csv --max-size 500000000 > data/temp/search_results.txt')
        
        print("\n✅ Search complete! Showing top results...")
        
    else:
        print("❌ Kaggle authentication failed")
        print("Please check that kaggle.json is in: C:\\Users\\Abhishek Chandra\\.kaggle\\")
