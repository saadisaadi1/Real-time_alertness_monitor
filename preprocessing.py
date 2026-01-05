import numpy as np
import os
from pathlib import Path
import pandas as pd
import cv2
import mediapipe as mp
import time
import datetime
import kagglehub

# Import MediaPipe utilities
from mp_utils import create_face_detector, crop_and_resize_face


def validate_video_label_matches(dataset_path):
    """
    Validate that all video files have matching labels in the CSV files
    before starting the actual processing.

    Args:
        dataset_path: Path to DAiSEE dataset

    Returns:
        tuple: (success: bool, match_stats: dict)
    """
    print("\n" + "🔍"*60)
    print("VALIDATING VIDEO-LABEL MATCHES BEFORE PROCESSING")
    print("🔍"*60 + "\n")

    # Check if DAiSEE subfolder exists
    daisee_subfolder = os.path.join(dataset_path, 'DAiSEE')
    if os.path.exists(daisee_subfolder):
        dataset_path = daisee_subfolder

    dataset_folder = os.path.join(dataset_path, 'DataSet')
    labels_folder = os.path.join(dataset_path, 'Labels')

    if not os.path.exists(dataset_folder):
        print(f"❌ ERROR: DataSet folder not found at {dataset_folder}")
        return False, {}

    # **NEW: Load ALL label CSV files in the Labels folder**
    print(f"Checking Labels folder: {labels_folder}")
    all_label_files = list(Path(labels_folder).glob("*.csv"))
    print(f"Found {len(all_label_files)} CSV files in Labels folder:")
    for lf in all_label_files:
        print(f"  - {lf.name}")

    # Load all labels into a single combined DataFrame
    all_labels_list = []
    for label_file in all_label_files:
        try:
            df_temp = pd.read_csv(label_file)
            df_temp.columns = df_temp.columns.str.strip()
            all_labels_list.append(df_temp)
            print(f"  Loaded {len(df_temp)} labels from {label_file.name}")
        except Exception as e:
            print(f"  ⚠️  Error loading {label_file.name}: {e}")

    if len(all_labels_list) == 0:
        print("❌ ERROR: No label files could be loaded!")
        return False, {}

    # Combine all labels
    combined_labels_df = pd.concat(all_labels_list, ignore_index=True)
    print(f"\n✅ Combined labels: {len(combined_labels_df)} total labels from all CSV files")
    print(f"Sample ClipIDs: {combined_labels_df['ClipID'].head(5).tolist()}")

    splits = ['Train', 'Validation', 'Test']
    overall_stats = {
        'total_videos': 0,
        'matched_videos': 0,
        'unmatched_videos': 0,
        'split_details': {},
        'all_unmatched_files': []
    }

    all_valid = True

    for split in splits:
        split_path = os.path.join(dataset_folder, split)

        if not os.path.exists(split_path):
            print(f"⚠️  {split} folder not found, skipping...")
            continue

        # Get video files
        video_files = (list(Path(split_path).rglob("*.avi")) +
                      list(Path(split_path).rglob("*.mp4")) +
                      list(Path(split_path).rglob("*.mkv")))

        print(f"\n{'='*60}")
        print(f"Validating {split}:")
        print(f"  Videos found: {len(video_files)}")
        print(f"  Searching in combined labels: {len(combined_labels_df)}")

        matched = 0
        unmatched = []
        unmatched_details = []

        for video_path in video_files:
            video_filename = video_path.name
            video_id = video_path.stem

            # **UPDATED: Search in the COMBINED labels DataFrame**
            label_row = combined_labels_df[combined_labels_df['ClipID'] == video_filename]

            if label_row.empty:
                label_row = combined_labels_df[combined_labels_df['ClipID'] == video_id]

            if label_row.empty:
                label_row = combined_labels_df[combined_labels_df['ClipID'].astype(str) == str(video_filename)]

            if label_row.empty:
                clean_video_id = str(video_id).replace('.avi', '').replace('.mp4', '')
                label_row = combined_labels_df[combined_labels_df['ClipID'].astype(str).str.replace('.avi', '').str.replace('.mp4', '') == clean_video_id]

            if label_row.empty:
                label_row = combined_labels_df[combined_labels_df['ClipID'].astype(str).str.lower() == str(video_filename).lower()]

            if not label_row.empty:
                matched += 1
            else:
                unmatched.append(video_filename)
                unmatched_details.append({
                    'split': split,
                    'filename': video_filename,
                    'video_id': video_id,
                    'full_path': str(video_path),
                    'parent_folder': video_path.parent.name,
                    'grandparent_folder': video_path.parent.parent.name
                })

        match_percentage = (matched / len(video_files) * 100) if len(video_files) > 0 else 0

        print(f"  ✅ Matched: {matched} ({match_percentage:.1f}%)")
        print(f"  ❌ Unmatched: {len(unmatched)} ({100-match_percentage:.1f}%)")

        # **NEW: Detailed analysis for unmatched videos**
        if len(unmatched) > 0:
            print(f"\n  🔍 ANALYZING {len(unmatched)} UNMATCHED VIDEOS IN {split}:")

            if len(unmatched) <= 10:
                print(f"  All unmatched videos: {unmatched}")
            else:
                print(f"  First 10 unmatched: {unmatched[:10]}")

            # Analyze first few unmatched videos
            for i, detail in enumerate(unmatched_details[:3]):
                print(f"\n  Example #{i+1}: {detail['filename']}")
                print(f"    video_id: '{detail['video_id']}'")
                print(f"    Folder: .../{ detail['grandparent_folder']}/{detail['parent_folder']}/")

                # Check if similar patterns exist in CSV
                video_prefix = detail['video_id'][:6]
                similar = combined_labels_df[combined_labels_df['ClipID'].str.startswith(video_prefix, na=False)]
                print(f"    Similar ClipIDs (prefix '{video_prefix}'): {len(similar)} found")
                if len(similar) > 0:
                    print(f"      Examples: {similar['ClipID'].head(3).tolist()}")

            # Show unique parent folders for unmatched
            unique_parents = set([d['grandparent_folder'] for d in unmatched_details])
            print(f"\n  Unique parent folders with unmatched videos: {sorted(list(unique_parents))}")

        overall_stats['total_videos'] += len(video_files)
        overall_stats['matched_videos'] += matched
        overall_stats['unmatched_videos'] += len(unmatched)
        overall_stats['all_unmatched_files'].extend(unmatched_details)
        overall_stats['split_details'][split] = {
            'total': len(video_files),
            'matched': matched,
            'unmatched': len(unmatched),
            'percentage': match_percentage,
            'unmatched_list': unmatched_details
        }

        if match_percentage < 99:
            all_valid = False
            print(f"  ⚠️  WARNING: Match rate below 99% for {split}!")

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL VALIDATION SUMMARY:")
    print(f"{'='*60}")
    print(f"Total videos across all splits: {overall_stats['total_videos']}")
    print(f"✅ Videos with labels: {overall_stats['matched_videos']} ({overall_stats['matched_videos']/overall_stats['total_videos']*100:.1f}%)")
    print(f"❌ Videos without labels: {overall_stats['unmatched_videos']} ({overall_stats['unmatched_videos']/overall_stats['total_videos']*100:.1f}%)")

    # Save unmatched videos to a file if any exist
    if len(overall_stats['all_unmatched_files']) > 0:
        output_dir = os.path.join(os.getcwd(), 'processed_daisee')
        os.makedirs(output_dir, exist_ok=True)

        unmatched_df = pd.DataFrame(overall_stats['all_unmatched_files'])
        unmatched_csv_path = os.path.join(output_dir, 'unmatched_videos.csv')
        unmatched_df.to_csv(unmatched_csv_path, index=False)

        print(f"\n📄 Unmatched videos list saved to: {unmatched_csv_path}")
        print(f"   Review this file to see which videos don't have labels.")

    # Store combined labels for later use
    overall_stats['combined_labels_df'] = combined_labels_df

    if overall_stats['matched_videos'] / overall_stats['total_videos'] >= 0.99:
        print(f"\n✅ VALIDATION PASSED - Excellent match rate!")
        print(f"   {overall_stats['matched_videos']} videos will be processed.")
        if overall_stats['unmatched_videos'] > 0:
            print(f"   {overall_stats['unmatched_videos']} videos will be skipped (no labels).")
    else:
        print(f"\n⚠️  VALIDATION WARNING - Some issues detected")
        print(f"   Processing will continue but some videos will be skipped.")

    print("="*60 + "\n")

    return all_valid, overall_stats


def process_daisee_dataset(dataset_path, detector, output_dir="processed_daisee", max_videos_per_split=None, combined_labels_df=None):
    """
    Process DAiSEE dataset: detect faces, crop, and resize to 224x224.
    Saves processed images to a separate directory without modifying the original.

    Args:
        dataset_path: Path to downloaded DAiSEE dataset (original kagglehub location)
        detector: MediaPipe face detector instance
        output_dir: Directory to save processed images (separate from original)
        max_videos_per_split: If set, only process this many videos per split (for testing)
        combined_labels_df: Combined DataFrame with all labels (if provided, use this instead of loading per-split CSVs)

    Returns:
        DataFrame with image paths and labels
    """
    start_time = time.time()

    # Create output directory in project folder, not in kagglehub cache
    output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if max_videos_per_split:
        print(f"\n⚠️  TESTING MODE: Processing {max_videos_per_split} videos from EACH split (mixed sampling)")
        print("="*60)

    print(f"Original dataset location: {dataset_path}")
    print(f"Processed data will be saved to: {output_dir}")

    # Check if DAiSEE subfolder exists (kagglehub structure)
    daisee_subfolder = os.path.join(dataset_path, 'DAiSEE')
    if os.path.exists(daisee_subfolder):
        print(f"Found DAiSEE subfolder, using: {daisee_subfolder}")
        dataset_path = daisee_subfolder

    dataset_folder = os.path.join(dataset_path, 'DataSet')
    labels_folder = os.path.join(dataset_path, 'Labels')

    if not os.path.exists(dataset_folder):
        print(f"ERROR: DataSet folder not found at {dataset_folder}")
        return pd.DataFrame(), output_dir

    # **NEW: Use combined labels if provided, otherwise load from Labels folder**
    if combined_labels_df is not None:
        print(f"\n✅ Using combined labels DataFrame with {len(combined_labels_df)} labels")
        labels_df = combined_labels_df
    else:
        # Load all label CSVs and combine them
        print(f"\nLoading labels from {labels_folder}...")
        all_label_files = list(Path(labels_folder).glob("*.csv"))
        all_labels_list = []
        for label_file in all_label_files:
            try:
                df_temp = pd.read_csv(label_file)
                df_temp.columns = df_temp.columns.str.strip()
                all_labels_list.append(df_temp)
            except Exception as e:
                print(f"  ⚠️  Error loading {label_file.name}: {e}")

        if len(all_labels_list) == 0:
            print("ERROR: No label files could be loaded!")
            return pd.DataFrame(), output_dir

        labels_df = pd.concat(all_labels_list, ignore_index=True)
        print(f"✅ Loaded {len(labels_df)} labels from all CSV files")

    # DAiSEE structure: DataSet/Train, DataSet/Validation, DataSet/Test
    splits = ['Train', 'Validation', 'Test']
    all_data = []

    for split in splits:
        split_start = time.time()
        split_path = os.path.join(dataset_folder, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found, skipping...")
            continue

        split_output = os.path.join(output_dir, split)
        os.makedirs(split_output, exist_ok=True)

        # Look for videos
        video_files = (list(Path(split_path).rglob("*.avi")) +
                      list(Path(split_path).rglob("*.mp4")) +
                      list(Path(split_path).rglob("*.mkv")))

        print(f"\n{'='*60}")
        print(f"Processing {split}:")
        print(f"  Total videos available: {len(video_files)}")

        if len(video_files) > 0:
            # **UPDATED: Sample videos from different parts of the dataset**
            if max_videos_per_split:
                # Take videos from beginning, middle, and end for diversity
                step = len(video_files) // max_videos_per_split if len(video_files) > max_videos_per_split else 1
                sampled_indices = [i * step for i in range(max_videos_per_split)]
                video_files = [video_files[i] for i in sampled_indices if i < len(video_files)]
                print(f"  ⚠️  Sampled {len(video_files)} videos from different parts (indices: {sampled_indices[:len(video_files)]})")
            else:
                print(f"  Processing all {len(video_files)} videos")

            print(f"  Starting processing...")

            for vid_idx, video_path in enumerate(video_files):
                print(f"  [{vid_idx+1}/{len(video_files)}] Processing: {video_path.name}")

                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print(f"    ❌ Could not open video")
                        continue

                    frame_count = 0
                    extracted_count = 0
                    video_id = video_path.stem
                    video_filename = video_path.name

                    # **UPDATED: Search in combined labels DataFrame**
                    video_labels = None
                    if labels_df is not None and 'ClipID' in labels_df.columns:
                        label_row = labels_df[labels_df['ClipID'] == video_filename]

                        if label_row.empty:
                            label_row = labels_df[labels_df['ClipID'] == video_id]

                        if label_row.empty:
                            label_row = labels_df[labels_df['ClipID'].astype(str) == str(video_filename)]

                        if label_row.empty:
                            clean_video_id = str(video_id).replace('.avi', '').replace('.mp4', '')
                            label_row = labels_df[labels_df['ClipID'].astype(str).str.replace('.avi', '').str.replace('.mp4', '') == clean_video_id]

                        if label_row.empty:
                            label_row = labels_df[labels_df['ClipID'].astype(str).str.lower() == str(video_filename).lower()]

                        if not label_row.empty:
                            video_labels = {
                                'Boredom': int(label_row.iloc[0]['Boredom']),
                                'Engagement': int(label_row.iloc[0]['Engagement']),
                                'Confusion': int(label_row.iloc[0]['Confusion']),
                                'Frustration': int(label_row.iloc[0]['Frustration'])
                            }
                            print(f"    ✅ Labels found: Boredom={video_labels['Boredom']}, Engagement={video_labels['Engagement']}, Confusion={video_labels['Confusion']}, Frustration={video_labels['Frustration']}")
                        else:
                            print(f"    ❌ No labels found - skipping video")
                            cap.release()
                            continue  # Skip videos without labels

                    if video_labels is None:
                        cap.release()
                        continue

                    # Extract frames at intervals
                    frame_interval = 30

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_count % frame_interval == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                            result = detector.detect(mp_image)

                            if result.detections:
                                best = max(result.detections, key=lambda d: d.categories[0].score)
                                bbox = best.bounding_box
                                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                                x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)

                                processed = crop_and_resize_face(frame, x1, y1, x2, y2, target_size=(224, 224))

                                output_filename = f"{video_id}_frame{frame_count:06d}.jpg"
                                output_path = os.path.join(split_output, output_filename)
                                cv2.imwrite(output_path, processed)

                                data_entry = {
                                    'original_path': str(video_path),
                                    'processed_path': output_path,
                                    'split': split,
                                    'filename': output_filename,
                                    'frame_number': frame_count,
                                    'video_id': video_id,
                                    'Boredom': video_labels['Boredom'],
                                    'Engagement': video_labels['Engagement'],
                                    'Confusion': video_labels['Confusion'],
                                    'Frustration': video_labels['Frustration']
                                }

                                all_data.append(data_entry)
                                extracted_count += 1

                        frame_count += 1

                    cap.release()
                    print(f"    ✅ Extracted {extracted_count} frames from {frame_count} total frames")

                except Exception as e:
                    print(f"    ❌ Error processing video: {e}")
                    continue

        split_elapsed = time.time() - split_start
        print(f"\n{split} completed in {split_elapsed:.1f}s ({split_elapsed/60:.1f} minutes)")
        print(f"Processed {len([d for d in all_data if d['split'] == split])} frames from {split}")

    # Create DataFrame and save
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TOTAL PROCESSING TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")

    if len(all_data) == 0:
        print("ERROR: No images were processed!")
        return pd.DataFrame(), output_dir

    df = pd.DataFrame(all_data)
    csv_path = os.path.join(output_dir, 'processed_dataset.csv')
    df.to_csv(csv_path, index=False)

    print(f"Processed {len(df)} images total")
    print(f"Train: {len(df[df['split']=='Train'])}")
    print(f"Validation: {len(df[df['split']=='Validation'])}")
    print(f"Test: {len(df[df['split']=='Test'])}")

    return df, output_dir


def load_processed_dataset(processed_dir="processed_daisee"):
    """
    Load already processed dataset from disk.

    Args:
        processed_dir: Directory containing processed images

    Returns:
        DataFrame with processed image paths and labels
    """
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    csv_path = os.path.join(processed_dir, 'processed_dataset.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed dataset not found at {csv_path}. Please run preprocessing first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded preprocessed dataset from: {csv_path}")
    print(f"Total images: {len(df)}")
    print(f"Train: {len(df[df['split']=='Train'])}")
    print(f"Validation: {len(df[df['split']=='Validation'])}")
    print(f"Test: {len(df[df['split']=='Test'])}")

    return df


def analyze_video_label_matching(dataset_path, split='Validation', sample_size=10):
    """
    Analyze how video files match with label CSV to debug mismatches.

    Args:
        dataset_path: Path to DAiSEE dataset
        split: Which split to analyze (Train/Validation/Test)
        sample_size: Number of videos to sample
    """
    print("\n" + "="*60)
    print(f"ANALYZING VIDEO-LABEL MATCHING FOR {split}")
    print("="*60)

    # Check if DAiSEE subfolder exists
    daisee_subfolder = os.path.join(dataset_path, 'DAiSEE')
    if os.path.exists(daisee_subfolder):
        dataset_path = daisee_subfolder

    dataset_folder = os.path.join(dataset_path, 'DataSet')
    labels_folder = os.path.join(dataset_path, 'Labels')

    split_path = os.path.join(dataset_folder, split)
    labels_csv = os.path.join(labels_folder, f"{split}Labels.csv")

    # Load labels
    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = labels_df.columns.str.strip()

    # Get video files
    video_files = (list(Path(split_path).rglob("*.avi")) +
                  list(Path(split_path).rglob("*.mp4")) +
                  list(Path(split_path).rglob("*.mkv")))

    print(f"\nFound {len(video_files)} video files in {split}")
    print(f"Found {len(labels_df)} labels in CSV")

    # Sample videos
    sample_videos = video_files[:sample_size] if len(video_files) > sample_size else video_files

    print(f"\nAnalyzing {len(sample_videos)} sample videos:")
    print("-" * 60)

    for video_path in sample_videos:
        video_filename = video_path.name
        video_id = video_path.stem
        video_parent = video_path.parent.name
        video_grandparent = video_path.parent.parent.name

        print(f"\nVideo: {video_filename}")
        print(f"  Full path: {video_path}")
        print(f"  video_id (stem): '{video_id}'")
        print(f"  Parent folder: '{video_parent}'")
        print(f"  Grandparent folder: '{video_grandparent}'")

        # Try to find matches
        exact_match = labels_df[labels_df['ClipID'] == video_filename]
        id_match = labels_df[labels_df['ClipID'] == video_id]
        parent_match = labels_df[labels_df['ClipID'].str.contains(video_parent, na=False)]

        print(f"  Exact filename match: {len(exact_match)} rows")
        print(f"  ID match: {len(id_match)} rows")
        print(f"  Parent folder match: {len(parent_match)} rows")

        if len(parent_match) > 0:
            print(f"    Sample parent matches: {parent_match['ClipID'].head(3).tolist()}")

    print("\n" + "-" * 60)
    print("\nSample ClipIDs from CSV:")
    print(labels_df['ClipID'].head(20).tolist())

    print("\n" + "-" * 60)
    print("\nSample video filenames:")
    for vf in video_files[:20]:
        print(f"  {vf.name} (parent: {vf.parent.name})")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Run preprocessing
    print("="*60)
    print("DAiSEE Dataset Preprocessing")
    print("="*60)

    # TESTING MODE: Set this to False to process all videos, or set to True for quick testing
    TESTING_MODE = False  # Changed to False for FULL processing
    MAX_VIDEOS = 5 if TESTING_MODE else None

    # **NEW: ANALYSIS MODE - Set to True to analyze matching before processing**
    ANALYSIS_MODE = False  # Changed to False - analysis confirmed matching works!

    if ANALYSIS_MODE:
        print("\n" + "🔍 "*20)
        print("ANALYSIS MODE - Analyzing video-label matching first")
        print("Set ANALYSIS_MODE = False to proceed with actual processing")
        print("🔍 "*20 + "\n")

        # Download original dataset
        download_path = kagglehub.dataset_download("olgaparfenova/daisee")
        print(f"\nDataset location: {download_path}")

        # Analyze each split
        for split in ['Train', 'Validation', 'Test']:
            try:
                analyze_video_label_matching(download_path, split=split, sample_size=10)
            except Exception as e:
                print(f"Error analyzing {split}: {e}")

        print("\n" + "="*60)
        print("Analysis complete!")
        print("Review the output above to understand the matching pattern.")
        print("Then set ANALYSIS_MODE = False to proceed with processing.")
        print("="*60)
        exit(0)

    if TESTING_MODE:
        print("\n" + "⚠️ "*20)
        print(f"TESTING MODE ENABLED - Processing {MAX_VIDEOS} videos from EACH split")
        print("Videos will be sampled from different parts of each split")
        print("Set TESTING_MODE = False in preprocessing.py to process full dataset")
        print("⚠️ "*20 + "\n")
    else:
        print("\n" + "🚀 "*20)
        print("FULL DATASET PROCESSING MODE")
        print(f"Processing ALL videos from the dataset")
        print("This will take approximately 15-20 minutes...")
        print("🚀 "*20 + "\n")

    # Download original dataset
    download_path = kagglehub.dataset_download("olgaparfenova/daisee")
    print(f"\nOriginal dataset downloaded to: {download_path}")
    print(f"Files: {os.listdir(download_path)}")

    # **NEW: VALIDATE VIDEO-LABEL MATCHES BEFORE PROCESSING**
    validation_success, match_stats = validate_video_label_matches(download_path)

    if match_stats['matched_videos'] == 0:
        print("\n❌ ERROR: No videos matched with labels!")
        print("Cannot proceed with processing. Please check the dataset structure and label files.")
        exit(1)

    # **UPDATED: Auto-proceed in testing mode, ask confirmation in full mode**
    if match_stats['unmatched_videos'] > 0 and not TESTING_MODE:
        print(f"\n⚠️  {match_stats['unmatched_videos']} videos will be skipped (no matching labels)")
        print(f"✅  {match_stats['matched_videos']} videos will be processed")
        response = input("\nDo you want to proceed with full processing? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Processing cancelled by user.")
            exit(1)
    elif match_stats['unmatched_videos'] > 0:
        print(f"\n⚠️  {match_stats['unmatched_videos']} videos will be skipped (no matching labels)")
        print("Auto-proceeding in testing mode...")

    # Create detector
    print("\nInitializing face detector...")
    detector = create_face_detector()

    # **NEW: Pass combined labels to processing function**
    print(f"\n{'='*60}")
    print("STARTING VIDEO PROCESSING")
    print(f"{'='*60}")
    processed_df, output_dir = process_daisee_dataset(
        download_path,
        detector,
        max_videos_per_split=MAX_VIDEOS,
        combined_labels_df=match_stats.get('combined_labels_df')
    )

    # Close detector
    detector.close()

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Original data remains at: {download_path}")
    print(f"Processed data saved to: {output_dir}")
    if TESTING_MODE:
        print("\n⚠️  Remember: This was a TEST RUN with limited videos")
        print("Set TESTING_MODE = False to process the full dataset")
    else:
        print(f"\n✅ Successfully processed {len(processed_df)} frames!")
        print("You can now run model1.py to train the engagement detection model.")
    print("="*60)
