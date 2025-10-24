import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# --- Configuration ---
INPUT_DATA_FILE = "data/training_data_v2.csv" # Start from the clean, original data
OUTPUT_DATA_FILE = "data/training_data_v2_augmented.csv" # Overwrite the old augmented file

# Column indices for features
NUM_NUMERICAL_FEATURES = 9
KEY_INDEX = 9
MODE_INDEX = 10
NUM_FEATURES = 11

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Loading base data from {INPUT_DATA_FILE}...")
    try:
        df = pd.read_csv(INPUT_DATA_FILE)
    except FileNotFoundError:
        logger.error(f"Error: {INPUT_DATA_FILE} not found. Please run convert_v1_to_v2.py first.")
        return

    logger.info(f"Loaded {len(df)} original triplets.")
    
    # Store all new synthetic triplets here
    new_triplets = []
    
    # Get all column names
    anchor_cols = [f'anchor_feat_{i}' for i in range(NUM_FEATURES)]
    positive_cols = [f'positive_feat_{i}' for i in range(NUM_FEATURES)]
    negative_cols = [f'negative_feat_{i}' for i in range(NUM_FEATURES)]
    all_cols = anchor_cols + positive_cols + negative_cols

    # --- 1. Get Unique Vectors ---
    all_original_vectors = pd.concat([
        df[anchor_cols].set_axis(anchor_cols, axis=1),
        df[positive_cols].set_axis(anchor_cols, axis=1),
        df[negative_cols].set_axis(anchor_cols, axis=1)
    ]).drop_duplicates().values

    logger.info(f"Found {len(all_original_vectors)} unique song vectors to augment.")

    # --- 2. Generate Comprehensive Synthetic Triplets ---
    logger.info("Generating comprehensive synthetic dataset (Categorical, Numerical, and Mixed)...")
    
    for vec in tqdm(all_original_vectors, desc="Augmenting Data (v3 logic)"):
        anchor = list(vec)
        
        # --- Type 1: Categorical Isolation ---
        # (Anchor, Anchor, Flipped_Mode)
        neg_mode = vec.copy()
        neg_mode[MODE_INDEX] = 1.0 - neg_mode[MODE_INDEX]
        new_triplets.append(anchor + anchor + list(neg_mode))
        
        # (Anchor, Anchor, Flipped_Key)
        neg_key = vec.copy()
        neg_key[KEY_INDEX] = (vec[KEY_INDEX] + 6) % 12 # Tritone shift
        new_triplets.append(anchor + anchor + list(neg_key))

        # --- Type 2: Numerical Isolation (Your "1000+" idea) ---
        for i in range(NUM_NUMERICAL_FEATURES):
            # (Anchor, Feature_High, Feature_Low)
            pos_num_high = vec.copy()
            pos_num_high[i] = min(1.0, vec[i] + 0.5) # Push up, cap at 1.0
            
            neg_num_low = vec.copy()
            neg_num_low[i] = max(0.0, vec[i] - 0.5) # Push down, cap at 0.0
            
            if pos_num_high[i] != neg_num_low[i]:
                new_triplets.append(anchor + list(pos_num_high) + list(neg_num_low))

        # --- Type 3: Mixed "Hard Negative" (Your "2000+" idea) ---
        # (Anchor, Anchor, Flipped_Mode_and_Energy)
        neg_mixed_1 = vec.copy()
        neg_mixed_1[MODE_INDEX] = 1.0 - neg_mixed_1[MODE_INDEX] # Flip mode
        neg_mixed_1[2] = 1.0 - neg_mixed_1[2] # Flip 'energy' (index 2)
        new_triplets.append(anchor + anchor + list(neg_mixed_1))
        
        # (Anchor, Anchor, Flipped_Key_and_Valence)
        neg_mixed_2 = vec.copy()
        neg_mixed_2[KEY_INDEX] = (vec[KEY_INDEX] + 6) % 12 # Flip key
        neg_mixed_2[7] = 1.0 - neg_mixed_2[7] # Flip 'valence' (index 7)
        new_triplets.append(anchor + anchor + list(neg_mixed_2))
        
    logger.info(f"Generated {len(new_triplets)} new synthetic triplets.")
    
    # --- 3. Combine and Save ---
    df_augmented = pd.concat([df, pd.DataFrame(new_triplets, columns=all_cols)], ignore_index=True)
    
    # Shuffle the dataset
    df_augmented = df_augmented.sample(frac=1).reset_index(drop=True)
    
    logger.info(f"Overwriting {OUTPUT_DATA_FILE} with {len(df_augmented)} total triplets...")
    df_augmented.to_csv(OUTPUT_DATA_FILE, index=False)
    logger.info("Augmentation complete!")

if __name__ == "__main__":
    main()