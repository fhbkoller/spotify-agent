# ml/data_pipeline/generate_triplets_v4.py
# (Based on V3's generate_triplets.py)

import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logging import logger
from ml.data_pipeline.triplet_miners import GenreTripletMiner, VibeTripletMiner

# --- V4.0 Configuration ---
LOOKUP_DIR = Path("ml/data/lookups")
FEATURES_FILE = LOOKUP_DIR / "features.parquet"
TAGS_ONEHOT_FILE = LOOKUP_DIR / "tags_onehot.parquet"
INVERTED_INDEX_FILE = LOOKUP_DIR / "inverted_tag_index.pkl"

# --- V4.0: New output format ---
OUTPUT_FILE = Path("ml/data/triplets_v4.parquet")
TARGET_TRIPLET_COUNT = 500_000

# Define the 11 raw feature columns from features.parquet
RAW_FEATURE_COLUMNS = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'speechiness', 'valence', 'tempo', 'loudness', # 0-8
    'key', 'mode' # 9, 10
]

def load_lookups():
    """
    Loads all Phase 1 artifacts required for V4 triplet generation.
    Specifically loads the 'genre_name' (string) column.
    """
    logger.info("Loading Phase 1 lookup artifacts...")
    if not (FEATURES_FILE.exists() and TAGS_ONEHOT_FILE.exists() and INVERTED_INDEX_FILE.exists()):
        logger.error("ERROR: Lookup files not found. Please run 'extract_audio_features.py' and 'build_jamendo_lookups.py' first.")
        return None, None, None, None
        
    features_df = pd.read_parquet(FEATURES_FILE)
    tags_onehot_df = pd.read_parquet(TAGS_ONEHOT_FILE)
    
    # --- V4.0 Check: Ensure raw features exist ---
    for col in RAW_FEATURE_COLUMNS:
        if col not in features_df.columns:
            logger.error(f"ERROR: Raw feature '{col}' not found in {FEATURES_FILE}.")
            return None, None, None, None

    # --- V4.0 Check: Ensure 'genre_name' string column exists ---
    if 'genre_name' not in tags_onehot_df.columns:
        logger.error(f"ERROR: 'genre_name' (string) column not found in {TAGS_ONEHOT_FILE}.")
        logger.error("Please ensure 'build_jamendo_lookups.py' (V4 version) was run successfully.")
        return None, None, None, None
    
    with open(INVERTED_INDEX_FILE, 'rb') as f:
        inverted_index = pickle.load(f)
    
    # --- V4.0: Create genre_lookup from the 'genre_name' string column ---
    genre_lookup = tags_onehot_df['genre_name']
    
    logger.info("Successfully loaded features, tags, inverted index, and V4 (string) genre lookup.")
    return features_df, tags_onehot_df, inverted_index, genre_lookup


def main():
    # 1. Load Phase 1 Artifacts
    features_df, tags_onehot_df, inverted_index, genre_lookup = load_lookups()
    if features_df is None:
        sys.exit(1)

    logger.info(f"Loaded {len(features_df)} tracks with 11 raw features.")
    
    # 3. Initialize Phase 2 Miners (These are compatible with V4)
    logger.info("Initializing triplet miners...")
    genre_miner = GenreTripletMiner(inverted_index, tags_onehot_df)
    vibe_miner = VibeTripletMiner(inverted_index, tags_onehot_df)
    miners = [genre_miner, vibe_miner]
    
    # 4. Generate Triplets
    logger.info(f"Generating {TARGET_TRIPLET_COUNT} V4 triplets (as dictionaries)...")
    
    all_valid_track_ids = list(tags_onehot_df.index)
    
    logger.info("Building track-to-tags lookup...")
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
        
        # --- V4.0: Get features as dictionaries ---
        try:
            # Get ANCHOR dict
            anchor_num_features = features_df.loc[anchor_id][RAW_FEATURE_COLUMNS].to_dict()
            anchor_genre = genre_lookup.loc[anchor_id] # Get the string, e.g., 'genre::rock'
            anchor_dict = {"genre": anchor_genre, **anchor_num_features} # Combine them
            
            # Get POSITIVE dict
            positive_num_features = features_df.loc[positive_id][RAW_FEATURE_COLUMNS].to_dict()
            positive_genre = genre_lookup.loc[positive_id]
            positive_dict = {"genre": positive_genre, **positive_num_features}
            
            # Get NEGATIVE dict
            negative_num_features = features_df.loc[negative_id][RAW_FEATURE_COLUMNS].to_dict()
            negative_genre = genre_lookup.loc[negative_id]
            negative_dict = {"genre": negative_genre, **negative_num_features}

        except KeyError as e:
            # A track_id was in the index but not in features_df or genre_lookup
            continue
            
        # 5. Store the triplet as a tuple of dictionaries
        all_triplets_data.append(
            (anchor_dict, positive_dict, negative_dict)
        )
        pbar.update(1)
        
    pbar.close()
    
    # 6. Save to Parquet
    logger.info(f"Generated {len(all_triplets_data)} triplets. Saving to Parquet...")
    
    # --- V4.0: Define columns for dictionaries ---
    all_cols = ['anchor', 'positive', 'negative']
    
    # Create final DataFrame from the list of tuples
    df_final = pd.DataFrame(all_triplets_data, columns=all_cols)
    
    # Shuffle the dataset
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    # Save to Parquet, which natively handles object/dict columns
    df_final.to_parquet(OUTPUT_FILE, index=False)
    
    logger.info("\n--- V4.0 Pipeline - Phase 1.B: Real-World Triplet Generation COMPLETE ---")
    logger.info(f"Final dataset (with feature dictionaries) saved to: {OUTPUT_FILE.resolve()}")
    logger.info("You can now run Phase 1.C (specificity_generator_v4.py).")


if __name__ == "__main__":
    main()