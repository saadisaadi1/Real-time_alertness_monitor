import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import load_processed_dataset

print("="*60)
print("DAiSEE Dataset Label Distribution")
print("="*60)

# Load preprocessed dataset
try:
    df = load_processed_dataset("processed_daisee")
except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("Please run: python preprocessing.py")
    exit(1)

print("\n" + "="*60)
print("Engagement Level Distribution (Detailed)")
print("="*60)

# Define engagement level names
engagement_levels = {
    0: 'Low',
    1: 'Medium-Low',
    2: 'Medium-High',
    3: 'High'
}

# Check if Engagement column exists
if 'Engagement' in df.columns:
    print("\nEngagement Level Counts (Full Dataset):")
    print("-" * 60)

    engagement_counts = df['Engagement'].value_counts().sort_index()
    total_samples = len(df)

    for level, label in engagement_levels.items():
        count = engagement_counts.get(level, 0)
        percentage = (count / total_samples) * 100
        bar = "█" * int(percentage / 2)  # Visual bar (each █ = 2%)
        print(f"  {label:<15} (Level {level}): {count:>6} samples ({percentage:>5.1f}%) {bar}")

    print(f"  {'-'*56}")
    print(f"  {'TOTAL':<15}        : {total_samples:>6} samples (100.0%)")

    # Additional statistics
    print(f"\n  Mean Engagement Level: {df['Engagement'].mean():.2f}")
    print(f"  Median Engagement Level: {df['Engagement'].median():.1f}")
    print(f"  Most Common Level: {engagement_levels[df['Engagement'].mode()[0]]} (Level {df['Engagement'].mode()[0]})")

else:
    print("❌ 'Engagement' column not found in dataset!")

print("\n" + "="*60)
print("Label Counts by Category")
print("="*60)

# Check if label columns exist
label_columns = ['Engagement', 'Boredom', 'Confusion', 'Frustration']
available_labels = [col for col in label_columns if col in df.columns]

if not available_labels:
    print("No label columns found in the dataset!")
    print(f"Available columns: {df.columns.tolist()}")
else:
    # Summary table
    print("\nSUMMARY:")
    print("-" * 60)
    print(f"{'Label Category':<20} {'Total Samples':<15} {'% of Dataset':<15}")
    print("-" * 60)

    for label_col in available_labels:
        total_samples = len(df[df[label_col].notna()])
        percentage = (total_samples / len(df)) * 100
        print(f"{label_col:<20} {total_samples:<15} {percentage:<15.1f}%")

    print("-" * 60)
    print(f"{'TOTAL DATASET':<20} {len(df):<15} {'100.0%':<15}")
    print("="*60)

    # Detailed distribution for each label
    label_map = {0: 'Very Low', 1: 'Low', 2: 'High', 3: 'Very High'}

    for label_col in available_labels:
        if label_col == 'Engagement':
            continue  # Already printed detailed engagement above

        print(f"\n{label_col} - Detailed Distribution:")
        print("-" * 60)
        value_counts = df[label_col].value_counts().sort_index()

        total_labeled = value_counts.sum()

        for value, count in value_counts.items():
            label_name = label_map.get(value, f"Unknown ({value})")
            percentage = (count / total_labeled) * 100
            percentage_total = (count / len(df)) * 100
            print(f"  {label_name:<15} (Value={value}): {count:>6} samples  "
                  f"({percentage:>5.1f}% of {label_col}, {percentage_total:>5.1f}% of total)")

        print(f"  {'-'*56}")
        print(f"  {'TOTAL':<15}          : {total_labeled:>6} samples")

# ORIGINAL Split distribution (from preprocessing)
print("\n" + "="*60)
print("ORIGINAL Split Distribution (from preprocessing.py)")
print("="*60)

if 'split' in df.columns:
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {split:<15}: {count:>6} samples ({percentage:>5.1f}%)")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<15}: {len(df):>6} samples (100.0%)")

    # Engagement distribution per original split
    print("\n" + "="*60)
    print("Engagement Distribution by ORIGINAL Split")
    print("="*60)

    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        print(f"\n{split}:")
        print("-" * 40)
        for level, label in engagement_levels.items():
            count = len(split_df[split_df['Engagement'] == level])
            percentage = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
            print(f"  {label:<15}: {count:>5} ({percentage:>5.1f}%)")

# NEW: 80/10/10 Split distribution (as used in model1.py)
print("\n" + "="*60)
print("80/10/10 TRAINING Split Distribution (as used in model1.py)")
print("="*60)

# Replicate the exact split from model1.py
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, temp_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"\nDataset split (80/10/10):")
print(f"  Total images: {len(df_shuffled)}")
print(f"  Train: {len(train_df)} ({len(train_df)/len(df_shuffled)*100:.1f}%)")
print(f"  Val: {len(val_df)} ({len(val_df)/len(df_shuffled)*100:.1f}%)")
print(f"  Test: {len(test_df)} ({len(test_df)/len(df_shuffled)*100:.1f}%)")

# Engagement distribution per 80/10/10 split
print("\n" + "="*60)
print("Engagement Distribution by 80/10/10 Split")
print("="*60)

for split_name, split_df in [('Train (80%)', train_df), ('Val (10%)', val_df), ('Test (10%)', test_df)]:
    print(f"\n{split_name}: {len(split_df)} samples")
    print("-" * 40)
    for level, label in engagement_levels.items():
        count = len(split_df[split_df['Engagement'] == level])
        percentage = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
        bar = "█" * int(percentage / 5)
        print(f"  {label:<15}: {count:>6} ({percentage:>5.1f}%) {bar}")

print("\n" + "="*60)
