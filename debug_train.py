"""
Debug script to check training data and diagnose issues
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("TRAINING DATA DIAGNOSTICS")
print("=" * 60)

# Load data
df = pd.read_csv('data/intents.csv')

print(f"\n1. Dataset Overview:")
print(f"   Total examples: {len(df)}")
print(f"   Columns: {list(df.columns)}")

print(f"\n2. Sample Data:")
print(df.head(10))

print(f"\n3. Check for Missing Values:")
print(df.isnull().sum())

# Parse multi-label intents
df['intent_list'] = df['intent'].apply(
    lambda x: [i.strip() for i in str(x).split('|')]
)

print(f"\n4. Intent Distribution:")
all_intents = []
for intents in df['intent_list']:
    all_intents.extend(intents)

intent_counts = pd.Series(all_intents).value_counts()
print(intent_counts)

print(f"\n5. Multi-label Examples:")
multi_label = df[df['intent'].str.contains('\|', regex=True)]
print(f"   Count: {len(multi_label)}")
if len(multi_label) > 0:
    print("\n   Examples:")
    for idx, row in multi_label.head(5).iterrows():
        print(f"   - {row['text']}")
        print(f"     Intents: {row['intent_list']}")

print(f"\n6. Single-label Examples:")
single_label = df[~df['intent'].str.contains('\|', regex=True)]
print(f"   Count: {len(single_label)}")

print(f"\n7. Text Length Statistics:")
df['text_length'] = df['text'].str.len()
print(f"   Min: {df['text_length'].min()}")
print(f"   Max: {df['text_length'].max()}")
print(f"   Average: {df['text_length'].mean():.1f}")

print(f"\n8. Checking for Duplicates:")
duplicates = df[df.duplicated(subset=['text'], keep=False)]
if len(duplicates) > 0:
    print(f"   ⚠️  Found {len(duplicates)} duplicate texts!")
    print(duplicates[['text', 'intent']])
else:
    print(f"   ✓ No duplicates found")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)

# Check if data is ready for training
issues = []

if len(df) < 20:
    issues.append("⚠️  Dataset too small (< 20 examples)")
    
if df.isnull().any().any():
    issues.append("⚠️  Missing values found")
    
if len(intent_counts) < 4:
    issues.append("⚠️  Not all intent categories present")

if len(issues) == 0:
    print("\n✅ Data looks good! Ready for training.")
    print("   Run: python src/train_intent.py")
else:
    print("\n❌ Issues found:")
    for issue in issues:
        print(f"   {issue}")