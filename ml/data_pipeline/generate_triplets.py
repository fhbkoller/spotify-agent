# ml/data_pipeline/generate_triplets.py (REVISED VERSION)

import pickle
import random
from pathlib import Path
import numpy as np # <-- NEW IMPORT
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

# These constants MUST be updated to reflect the 2D key encoding:
NUM_NUMERICAL_FEATURES_RAW = 9   # acousticness through loudness/tempo
NUM_FEATURES_KEY_ENCODED = 2     # key_sin, key_cos
NUM_CATEGORICAL_RAW = 1          # mode
NUM_TOTAL_FEATURES = 12          # 9 + 2 + 1 = 12

# Note: The raw feature files have 11 columns (9 numerical, 2 categorical: key, mode).
# After encoding, we have 12 (9 numerical, 2 key-encoded, 1 mode).

def load_lookups():
    # (Unchanged from previous version)
    # ...
    # (function implementation remains the same)
    # ...
    
    # 1. Load Phase 1 Artifacts
    print("Loading Phase 1 lookup artifacts...")
    if not (FEATURES_FILE.exists() and TAGS_ONEHOT_FILE.exists() and INVERTED_INDEX_FILE.exists()):
        print("ERROR: Lookup files not found. Please run `build_jamendo_lookups.py` first.")
        return None, None, None
        
    features_df = pd.read_parquet(FEATURES_FILE)
    with open(INVERTED_INDEX_FILE, 'rb') as f:
        inverted_index = pickle.load(f)
    tags_onehot_df = pd.read_parquet(TAGS_ONEHOT_FILE)
    
    return features_df, tags_onehot_df, inverted_index

def scale_and_encode_features(features_df: pd.DataFrame):
    """
    Applies MinMaxScaler to 9 numerical features and Cyclical Encoding (sin/cos)
    to the 'key' feature (0-11).
    """
    print("Applying MinMaxScaler and Cyclical Encoding...")
    
    # Identify feature sets based on the raw 11 columns
    numerical_cols = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'speechiness', 'valence', 'loudness', 'tempo'
    ]
    key_col = 'key'
    mode_col = 'mode'
    
    if features_df.shape[1] != 11:
         raise Exception(f"Input DataFrame must have 11 columns (found {features_df.shape[1]}).")
        
    # --- 1. MIN/MAX SCALING (9 features) ---
    scaler = MinMaxScaler()
    scaled_numerical_df = pd.DataFrame(
        scaler.fit_transform(features_df[numerical_cols]),
        columns=numerical_cols,
        index=features_df.index
    )
    
    # --- 2. CYCLICAL ENCODING (Key) ---
    # The key (0-11) is mapped to a 360-degree circle.
    # We use 12 for the period since the range is 0 to 11 (12 possible values)
    period = 12 
    features_df['key_sin'] = np.sin(2 * np.pi * features_df[key_col] / period)
    features_df['key_cos'] = np.cos(2 * np.pi * features_df[key_col] / period)
    
    # --- 3. RE-COMBINE AND ORDER ---
    
    # Start with the scaled numericals
    scaled_features_df = scaled_numerical_df.copy()
    
    # Add the 2 new key-encoded features (float, scaled -1 to 1 automatically)
    scaled_features_df['key_sin'] = features_df['key_sin']
    scaled_features_df['key_cos'] = features_df['key_cos']
    
    # Add the raw 'mode' feature (0 or 1, which is already a numerical category)
    scaled_features_df['mode'] = features_df[mode_col]
    
    # Define the final column order (9 numerical, 2 key, 1 mode)
    final_cols = numerical_cols + ['key_sin', 'key_cos'] + [mode_col]
    scaled_features_df = scaled_features_df[final_cols]
    
    if scaled_features_df.shape[1] != NUM_TOTAL_FEATURES:
        raise Exception(f"Final feature count mismatch! Expected {NUM_TOTAL_FEATURES}, got {scaled_features_df.shape[1]}.")
        
    print(f"Feature processing complete. Total features per track: {NUM_TOTAL_FEATURES}")
    return scaled_features_df


def main():
    # 1. Load Phase 1 Artifacts
    features_df, tags_onehot_df, inverted_index = load_lookups()
    if features_df is None:
        return

    # 2. Scale and Encode features
    scaled_features_df = scale_and_encode_features(features_df)
    
    # 3. Initialize Phase 2 Miners (Unchanged)
    print("Initializing triplet miners...")
    genre_miner = GenreTripletMiner(inverted_index, tags_onehot_df)
    vibe_miner = VibeTripletMiner(inverted_index, tags_onehot_df)
    miners = [genre_miner, vibe_miner]
    
    # ... (Rest of the triplet generation loop remains the same) ...
    
    # 4. Generate Triplets
    print(f"Generating {TARGET_TRIPLET_COUNT} triplets...")
    
    all_valid_track_ids = list(tags_onehot_df.index)
    
    # This track-to-tags lookup is critical for mining. It uses normalized tags.
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
        
        # 4. Get the 12 features for A, P, N
        try:
            # We use .loc[] and .values to get the 12-element numpy array
            anchor_features = scaled_features_df.loc[anchor_id].values
            positive_features = scaled_features_df.loc[positive_id].values
            negative_features = scaled_features_df.loc[negative_id].values
        except KeyError:
            continue
            
        # 5. Store the triplet
        all_triplets_data.append(
            list(anchor_features) + list(positive_features) + list(negative_features)
        )
        pbar.update(1)
        
    pbar.close()
    
    # 6. Save to CSV
    print(f"Generated {len(all_triplets_data)} triplets. Saving to CSV...")
    
    # Define column names based on the NEW 12 features
    anchor_cols = [f'anchor_feat_{i}' for i in range(NUM_TOTAL_FEATURES)]
    positive_cols = [f'positive_feat_{i}' for i in range(NUM_TOTAL_FEATURES)]
    negative_cols = [f'negative_feat_{i}' for i in range(NUM_TOTAL_FEATURES)]
    all_cols = anchor_cols + positive_cols + negative_cols
    
    # Create final DataFrame
    df_final = pd.DataFrame(all_triplets_data, columns=all_cols)
    
    # Shuffle the dataset
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    # Save
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- Phase 3: Triplet Generation COMPLETE ---")
    print(f"Final dataset saved to: {OUTPUT_FILE.resolve()}")
    print(f"Dataset has {NUM_TOTAL_FEATURES} features per track (3 x 12 = 36 columns total).")


if __name__ == "__main__":
    main()