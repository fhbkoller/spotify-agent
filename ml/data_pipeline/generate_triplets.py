# ml/data_pipeline/generate_triplets.py (CORRECTED)

import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our miners from Phase 2
from triplet_miners import GenreTripletMiner, VibeTripletMiner

# --- Configuration ---
LOOKUP_DIR = Path("ml/data/lookups")
FEATURES_FILE = LOOKUP_DIR / "features.parquet"
TAGS_ONEHOT_FILE = LOOKUP_DIR / "tags_onehot.parquet"
INVERTED_INDEX_FILE = LOOKUP_DIR / "inverted_tag_index.pkl"

OUTPUT_FILE = Path("ml/data/triplets.csv")
TARGET_TRIPLET_COUNT = 500_000 # From the plan

# --- V3.0: We save the 11 RAW features ---
NUM_TOTAL_FEATURES = 11 # 9 numericals + key + mode

# Define the 11 raw feature columns in the order our V3 dataset expects
RAW_FEATURE_COLUMNS = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'speechiness', 'valence', 'tempo', 'loudness', # 0-8
    'key', 'mode' # 9, 10
]

def load_lookups():
    # 1. Load Phase 1 Artifacts
    print("Loading Phase 1 lookup artifacts...")
    if not (FEATURES_FILE.exists() and TAGS_ONEHOT_FILE.exists() and INVERTED_INDEX_FILE.exists()):
        print("ERROR: Lookup files not found. Please run `build_jamendo_lookups.py` first.")
        return None, None, None, None
        
    features_df = pd.read_parquet(FEATURES_FILE)
    tags_onehot_df = pd.read_parquet(TAGS_ONEHOT_FILE)
    
    # --- V3.0 FIX: Check for raw features in features_df ---
    for col in RAW_FEATURE_COLUMNS:
        if col not in features_df.columns:
            print(f"ERROR: Raw feature '{col}' not found in {FEATURES_FILE}.")
            return None, None, None, None

    # --- V3.0 FIX: Check for genre_label in tags_onehot_df ---
    if 'genre_label' not in tags_onehot_df.columns:
        print(f"ERROR: 'genre_label' not found in {TAGS_ONEHOT_FILE}.")
        print("Please ensure `build_jamendo_lookups.py` saved the genre label there.")
        return None, None, None, None
    
    with open(INVERTED_INDEX_FILE, 'rb') as f:
        inverted_index = pickle.load(f)
    
    # --- V3.0 FIX: Create genre_lookup from tags_onehot_df ---
    genre_lookup = tags_onehot_df['genre_label']
    
    print("Successfully loaded features, tags, inverted index, and genre lookup.")
    # Return all 4 items
    return features_df, tags_onehot_df, inverted_index, genre_lookup


def main():
    # 1. Load Phase 1 Artifacts
    # --- V3.0 FIX: Load 4 items ---
    features_df, tags_onehot_df, inverted_index, genre_lookup = load_lookups()
    if features_df is None:
        return

    # 2. We no longer scale features here. We will save the raw features.
    print(f"Loaded {len(features_df)} tracks with 11 raw features.")
    
    # 3. Initialize Phase 2 Miners
    print("Initializing triplet miners...")
    genre_miner = GenreTripletMiner(inverted_index, tags_onehot_df)
    vibe_miner = VibeTripletMiner(inverted_index, tags_onehot_df)
    miners = [genre_miner, vibe_miner]
    
    # 4. Generate Triplets
    print(f"Generating {TARGET_TRIPLET_COUNT} triplets...")
    
    all_valid_track_ids = list(tags_onehot_df.index)
    
    print("Building track-to-tags lookup...")
    track_to_tags = {}
    for tag, tracks in inverted_index.items():
        for track in tracks:
            track_to_tags.setdefault(track, set()).add(tag)
    
    all_triplets_data = []
    
    pbar = tqdm(total=TARGET_TRIPLET_COUNT)
    while len(all_triplets_data) < TARGET_TRIPLET_COUNT:
        # 1. Pick a random anchor track
        anchor_id = random.choice(all_valid_track_ids)
        anchor_tags = track_to_tags.get(anchor_id)
        if not anchor_tags:
            continue

        # 2. Pick a random miner
        miner = random.choice(miners)
        
        # 3. Try to mine a triplet
        try:
            result = miner.mine(anchor_id, anchor_tags)
        except Exception:
            continue
            
        if not result:
            continue

        positive_id, negative_id = result
        
        # 4. Get the 11 RAW features for A, P, N + genre for A
        try:
            # --- V3.0 FIX: Get features from features_df ---
            anchor_features = features_df.loc[anchor_id][RAW_FEATURE_COLUMNS].values
            positive_features = features_df.loc[positive_id][RAW_FEATURE_COLUMNS].values
            negative_features = features_df.loc[negative_id][RAW_FEATURE_COLUMNS].values
            
            # --- V3.0 FIX: Get the anchor genre label from genre_lookup ---
            anchor_genre_label = genre_lookup.loc[anchor_id]

        except KeyError as e:
            # A track_id was in the index but not in the features_df or genre_lookup
            # print(f"KeyError: {e}") # Uncomment for debugging
            continue
            
        # 5. Store the triplet (11 + 11 + 11 + 1 = 34 columns)
        all_triplets_data.append(
            list(anchor_features) + list(positive_features) + list(negative_features) + [int(anchor_genre_label)]
        )
        pbar.update(1)
        
    pbar.close()
    
    # 6. Save to CSV
    print(f"Generated {len(all_triplets_data)} triplets. Saving to CSV...")
    
    # --- V3.0: Define columns for 11 raw features + 1 label ---
    anchor_cols = [f'anchor_feat_{i}' for i in range(NUM_TOTAL_FEATURES)]
    positive_cols = [f'positive_feat_{i}' for i in range(NUM_TOTAL_FEATURES)]
    negative_cols = [f'negative_feat_{i}' for i in range(NUM_TOTAL_FEATURES)]
    
    # Add the crucial genre label column
    all_cols = anchor_cols + positive_cols + negative_cols + ['anchor_genre_label']
    
    # Create final DataFrame
    df_final = pd.DataFrame(all_triplets_data, columns=all_cols)
    
    # Shuffle the dataset
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    # Save
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- Phase 3: V2 Triplet Generation COMPLETE ---")
    print(f"Final dataset saved to: {OUTPUT_FILE.resolve()}")
    print(f"Dataset has {NUM_TOTAL_FEATURES} raw features per track (3 x 11 = 33 columns) + 1 'anchor_genre_label' column.")


if __name__ == "__main__":
    main()