import pandas as pd
import os

print("="*60)
print("DAiSEE Dataset Label Distribution")
print("="*60)

# Load preprocessed dataset
csv_path = os.path.join("processed_daisee", "dataset.csv")
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded preprocessed dataset from: {csv_path}")
    print(f"Total images: {len(df)}")
except FileNotFoundError:
    print(f"\nError: {csv_path} not found!")
    print("Please run: python preprocessing.py")
    exit(1)

# Define engagement level names
engagement_levels = {
    0: 'Low',
    1: 'Medium-Low',
    2: 'Medium-High',
    3: 'High'
}

print("\n" + "="*60)
print("Overall Dataset Statistics")
print("="*60)
print(f"Total samples: {len(df)}")
if 'split' in df.columns:
    for split in ['Train', 'Validation', 'Test']:
        count = len(df[df['split'] == split])
        print(f"{split}: {count} samples ({count/len(df)*100:.1f}%)")

print("\n" + "="*60)
print("CLASS DISTRIBUTION BY SPLIT")
print("="*60)

if 'split' in df.columns and 'engagement' in df.columns:
    for split in ['Train', 'Validation', 'Test']:
        split_df = df[df['split'] == split]
        if len(split_df) == 0:
            continue

        print(f"\n{split.upper()}: {len(split_df)} total samples")
        print("-" * 60)

        for level in sorted(engagement_levels.keys()):
            label = engagement_levels[level]
            count = len(split_df[split_df['engagement'] == level])
            percentage = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"  {label:<15} (Level {level}): {count:>6} samples ({percentage:>5.1f}%) {bar}")

print("\n" + "="*60)
print("OVERALL CLASS DISTRIBUTION")
print("="*60)

print(f"\n{'Class':<15} {'Count':<10} {'Percentage':<12} {'Distribution'}")
print("-" * 60)

total_samples = len(df)
for level in sorted(engagement_levels.keys()):
    label = engagement_levels[level]
    count = len(df[df['engagement'] == level])
    percentage = (count / total_samples) * 100
    bar = "█" * int(percentage / 2)
    print(f"{label:<15} {count:<10} {percentage:>5.1f}%      {bar}")

print(f"\n{'TOTAL':<15} {total_samples:<10} 100.0%")

print("\n" + "="*60)
print("CROSS-SPLIT CLASS COMPARISON")
print("="*60)

print(f"\n{'Class':<15} {'Train':<12} {'Validation':<12} {'Test':<12} {'Total':<12}")
print("-" * 75)

for level in sorted(engagement_levels.keys()):
    label = engagement_levels[level]
    train_count = len(df[(df['split'] == 'Train') & (df['engagement'] == level)])
    val_count = len(df[(df['split'] == 'Validation') & (df['engagement'] == level)])
    test_count = len(df[(df['split'] == 'Test') & (df['engagement'] == level)])
    total_count = train_count + val_count + test_count

    print(f"{label:<15} {train_count:<12} {val_count:<12} {test_count:<12} {total_count:<12}")

print("-" * 75)
train_total = len(df[df['split'] == 'Train'])
val_total = len(df[df['split'] == 'Validation'])
test_total = len(df[df['split'] == 'Test'])
print(f"{'TOTAL':<15} {train_total:<12} {val_total:<12} {test_total:<12} {len(df):<12}")

print("\n" + "="*60)
print("CLASS BALANCE ANALYSIS")
print("="*60)

print("\nClass distribution percentages within each split:")
for split in ['Train', 'Validation', 'Test']:
    split_df = df[df['split'] == split]
    if len(split_df) == 0:
        continue
    print(f"\n{split}:")
    for level in sorted(engagement_levels.keys()):
        label = engagement_levels[level]
        count = len(split_df[split_df['engagement'] == level])
        percentage = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
        print(f"  {label:<15}: {percentage:>5.1f}%")

print("\n" + "="*60)

