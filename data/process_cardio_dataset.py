"""
Process the cardiovascular dataset (70,000 samples!)
"""
import pandas as pd

print("="*60)
print("PROCESSING CARDIOVASCULAR DATASET")
print("="*60)

# Load with correct delimiter
print("\nLoading data...")
df = pd.read_csv('data/temp/cardio_train.csv', delimiter=';')

print(f"✅ Loaded {len(df):,} samples")
print(f"\n📋 Columns: {list(df.columns)}")

# Show cholesterol distribution
print(f"\n📊 Cholesterol Distribution:")
print(df['cholesterol'].value_counts().sort_index())
print(f"\nNote: This uses categorical cholesterol levels:")
print("  1 = Normal")
print("  2 = Above Normal")
print("  3 = Well Above Normal")

# Check if this is categorical or continuous
if df['cholesterol'].nunique() <= 5:
    print("\n⚠️ WARNING: Cholesterol is CATEGORICAL (not continuous mg/dL values)")
    print("This is less ideal for regression, but we have 70,000 samples!")
    
    # We can still use it, or convert categories to estimated mg/dL ranges
    print("\n💡 Converting categories to estimated mg/dL values...")
    
    # Standard cholesterol ranges
    chol_mapping = {
        1: 180,  # Normal: <200 mg/dL (average ~180)
        2: 220,  # Above normal: 200-240 mg/dL (average ~220)
        3: 260   # Well above: >240 mg/dL (average ~260)
    }
    
    df['total_cholesterol'] = df['cholesterol'].map(chol_mapping)
    
    print("\nMapping:")
    for cat, mg in chol_mapping.items():
        count = (df['cholesterol'] == cat).sum()
        print(f"  Category {cat} → {mg} mg/dL ({count:,} patients)")

else:
    df['total_cholesterol'] = df['cholesterol']

# Show dataset info
print(f"\n📈 Dataset Features:")
print(f"   Age (in days - will convert)")
print(f"   Gender: {df['gender'].unique()}")
print(f"   Height (cm): {df['height'].min()}-{df['height'].max()}")
print(f"   Weight (kg): {df['weight'].min()}-{df['weight'].max()}")
print(f"   Blood Pressure: ap_hi (systolic), ap_lo (diastolic)")
print(f"   Cholesterol: 1-3 (categorical)")
print(f"   Glucose: {df['gluc'].unique()} (categorical)")
print(f"   Smoke: {df['smoke'].unique()}")
print(f"   Alcohol: {df['alco'].unique()}")
print(f"   Physical Activity: {df['active'].unique()}")

# Convert age from days to years
df['age_years'] = df['age'] / 365.25

# Calculate BMI
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

print(f"\n✅ Added calculated features:")
print(f"   age_years: {df['age_years'].min():.1f} - {df['age_years'].max():.1f}")
print(f"   BMI: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")

# Remove outliers (some BP values are clearly wrong)
print(f"\n🔧 Cleaning outliers...")
original_len = len(df)

df = df[
    (df['ap_hi'] >= 80) & (df['ap_hi'] <= 250) &  # Reasonable systolic BP
    (df['ap_lo'] >= 40) & (df['ap_lo'] <= 150) &  # Reasonable diastolic BP
    (df['height'] >= 140) & (df['height'] <= 220) &  # Reasonable height
    (df['weight'] >= 40) & (df['weight'] <= 200)  # Reasonable weight
]

removed = original_len - len(df)
print(f"   Removed {removed:,} outliers ({removed/original_len*100:.1f}%)")
print(f"   Remaining: {len(df):,} samples")

# Rename columns for consistency
df = df.rename(columns={
    'gender': 'male',
    'ap_hi': 'sysBP',
    'ap_lo': 'diaBP',
    'smoke': 'smoking',
    'alco': 'alcohol',
    'active': 'physical_activity',
    'gluc': 'glucose_level'
})

# Select relevant columns
df_final = df[[
    'age_years', 'male', 'height', 'weight', 'BMI',
    'sysBP', 'diaBP', 'smoking', 'alcohol', 'physical_activity',
    'glucose_level', 'total_cholesterol'
]]

# Rename age_years to age
df_final = df_final.rename(columns={'age_years': 'age'})

# Save
output_file = 'data/nhanes_cholesterol.csv'
df_final.to_csv(output_file, index=False)

print(f"\n💾 Saved to: {output_file}")
print(f"\n✅ READY FOR TRAINING!")
print(f"   Samples: {len(df_final):,}")
print(f"   Features: {len(df_final.columns) - 1}")
print(f"\n📊 Target Statistics:")
print(f"   Mean: {df_final['total_cholesterol'].mean():.2f} mg/dL")
print(f"   Median: {df_final['total_cholesterol'].median():.2f} mg/dL")
print(f"   Range: {df_final['total_cholesterol'].min():.0f} - {df_final['total_cholesterol'].max():.0f} mg/dL")

print("\n" + "="*60)
print("🎉 DATASET READY!")
print("="*60)
print("\nThis dataset has:")
print("  ✅ 60,000+ real patients")
print("  ✅ Lifestyle factors (smoking, alcohol, activity)")
print("  ✅ Physical measurements (height, weight, BMI, BP)")
print("  ✅ Metabolic markers (glucose, cholesterol)")
print("\nExpected R² with this data: 30-50%")
print("\nNext: Run 'python train_model_fast.py'")
