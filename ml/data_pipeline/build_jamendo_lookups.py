# ml/data_pipeline/build_jamendo_lookups.py

import pandas as pd
from pathlib import Path
import ast
import pickle
from scipy.sparse import lil_matrix, save_npz, csr_matrix
from tqdm import tqdm
import traceback
import sys
import numpy as np
import os

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logging import logger

# --- Configuration ---
LOOKUP_DIR = Path("ml/data/lookups")
FEATURES_FILE = LOOKUP_DIR / "features.parquet"
TAGS_ONEHOT_FILE = LOOKUP_DIR / "tags_onehot.parquet"
INVERTED_INDEX_FILE = LOOKUP_DIR / "inverted_tag_index.pkl"

# Direct URLs to the raw metadata files
TRAIN_TSV_URL = "https://huggingface.co/datasets/rkstgr/mtg-jamendo/raw/main/train.tsv"
VALID_TSV_URL = "https://huggingface.co/datasets/rkstgr/mtg-jamendo/raw/main/valid.tsv"

def safe_eval(tag_string: str):
    """
    Safely evaluate a string representation of a list (e.g., "['tag1', 'tag2']").
    Uses ast.literal_eval for security.
    """
    try:
        return ast.literal_eval(tag_string)
    except (ValueError, SyntaxError, TypeError):
        return [] # Return empty list on bad data (like np.nan)

def run():
    logger.info("--- V4.0 Pipeline - Phase 1 (Part B): Generating Tag Lookups ---")
    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load features.parquet to get all valid track_ids ---
    logger.info(f"Loading features from {FEATURES_FILE} to get track IDs...")
    try:
        features_df = pd.read_parquet(FEATURES_FILE)
    except FileNotFoundError:
        logger.error(f"FATAL ERROR: {FEATURES_FILE} not found.")
        logger.error("Please run 'extract_audio_features.py' (Phase 1A) first.")
        sys.exit(1)
        
    all_tracks_in_features = set(features_df.index.astype(str))
    logger.info(f"Found {len(all_tracks_in_features)} processed tracks.")
    if not all_tracks_in_features:
        logger.error("FATAL ERROR: No tracks found in features.parquet.")
        sys.exit(1)

    # --- 2. Download and Load Tag Data directly ---
    logger.info("Starting Phase 1b: Downloading tag lookups from Hugging Face URLs...")
    try:
        logger.info(f"Downloading {TRAIN_TSV_URL}...")
        df_train = pd.read_csv(TRAIN_TSV_URL, sep="\t")
        
        logger.info(f"Downloading {VALID_TSV_URL}...")
        df_valid = pd.read_csv(VALID_TSV_URL, sep="\t")
        
        df_all_tags = pd.concat([df_train, df_valid], ignore_index=True)
        logger.info(f"Successfully downloaded and combined {len(df_all_tags)} total metadata records.")

    except Exception as e:
        logger.error(f"ERROR: Failed to download or parse .tsv files: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 3. Process Tags ---
    logger.info("Parsing tag strings (this may take a minute)...")
    
    df_all_tags["genres_list"] = df_all_tags["genres"].apply(safe_eval)
    df_all_tags["instruments_list"] = df_all_tags["instruments"].apply(safe_eval)
    df_all_tags["moods_list"] = df_all_tags["moods"].apply(safe_eval)

    logger.info("Cross-referencing tags with processed audio tracks...")
    track_to_tags = {}
    all_tags = set()
    inverted_tag_index = {} # tag -> set(track_id)

    for _, row in tqdm(df_all_tags.iterrows(), total=len(df_all_tags), desc="Building Tag Maps"):
        track_id = str(row['id'])
        
        if track_id in all_tracks_in_features:
            
            cleaned_tags = set()
            for tag in row['genres_list']:
                cleaned_tags.add(f"genre::{tag.lower().replace(' ', '-')}")
            for tag in row['instruments_list']:
                cleaned_tags.add(f"instrument::{tag.lower().replace(' ', '-')}")
            for tag in row['moods_list']:
                cleaned_tags.add(f"mood::{tag.lower().replace(' ', '-')}")
            
            if not cleaned_tags:
                continue
                
            track_to_tags[track_id] = cleaned_tags
            all_tags.update(cleaned_tags)
            
            for tag in cleaned_tags:
                if tag not in inverted_tag_index:
                    inverted_tag_index[tag] = set()
                inverted_tag_index[tag].add(track_id)

    if not track_to_tags:
        logger.error("ERROR: No matching tracks found between features.parquet and tag files.")
        sys.exit(1)
        
    logger.info(f"Found {len(track_to_tags)} matching tracks with {len(all_tags)} unique *categorized* tags.")
    
    # --- 4. Build One-Hot Sparse Matrix ---
    logger.info("Building sparse one-hot tag matrix...")
    
    all_tags_list = sorted(list(all_tags))
    tag_to_int = {tag: i for i, tag in enumerate(all_tags_list)}
    
    final_track_ids = sorted(list(track_to_tags.keys()))
    track_to_int = {track_id: i for i, track_id in enumerate(final_track_ids)}
    
    num_tracks = len(final_track_ids)
    num_tags = len(all_tags_list)
    
    sparse_matrix = lil_matrix((num_tracks, num_tags), dtype=int)
    
    for track_id, tags in tqdm(track_to_tags.items(), desc="Populating Sparse Matrix"):
        row_idx = track_to_int[track_id]
        for tag in tags:
            col_idx = tag_to_int[tag]
            sparse_matrix[row_idx, col_idx] = 1
            
    sparse_matrix_csr = sparse_matrix.tocsr()
    
    one_hot_df = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix_csr,
        index=final_track_ids,
        columns=all_tags_list
    )
    one_hot_df.index.name = 'track_id'

    # --- 5. V4.0 STEP: Create the single-label 'genre_label' (INT) and 'genre_name' (STRING) columns ---
    logger.info("Creating single-label 'genre_label' and 'genre_name' for V4 model...")
    
    # 5a. Get all genre columns, sorted alphabetically
    genre_cols = sorted([col for col in one_hot_df.columns if col.startswith('genre::')])
    
    if not genre_cols:
        logger.error("ERROR: No 'genre::' columns found. Cannot create 'genre_label'.")
        sys.exit(1)
        
    logger.info(f"Found {len(genre_cols)} unique genres. This will be our NUM_GENRES.")
    
    # *** V4.0 NEW (Task 1.A) ***
    # Create the lookup map from integer index -> string name
    genre_name_lookup = {i: name for i, name in enumerate(genre_cols)}
    logger.info(f"Example: 0 -> '{genre_name_lookup[0]}', 1 -> '{genre_name_lookup[1]}', etc.")
    
    # 5b. Get the subset of the DataFrame with only genre columns
    genre_df = one_hot_df[genre_cols]
    
    # 5c. Use argmax to find the integer index of the first '1' in each row.
    genre_indices = genre_df.to_numpy().argmax(axis=1)
    
    # 5d. Check for tracks that have NO genre at all (sum of 1s is 0)
    has_no_genre = (genre_df.sum(axis=1) == 0)
    
    # 5e. Create the new columns in the main DataFrame
    # Default to the argmax index
    one_hot_df['genre_label'] = genre_indices
    # Set tracks with no genre to -1 (we can filter these out later)
    one_hot_df.loc[has_no_genre, 'genre_label'] = -1
    
    # *** V4.0 NEW (Task 1.A) ***
    # Map the integer label back to the string name for our new V4 token
    one_hot_df['genre_name'] = one_hot_df['genre_label'].map(genre_name_lookup).fillna('genre::unknown')
    # Explicitly set no-genre tracks to 'genre::unknown'
    one_hot_df.loc[has_no_genre, 'genre_name'] = 'genre::unknown'
    
    logger.info("Successfully created 'genre_label' (int) and 'genre_name' (str) columns.")

    # --- 6. Save Artifacts ---
    logger.info("Saving lookup artifacts...")
    
    # 6a. Save Inverted Tag Index
    with open(INVERTED_INDEX_FILE, 'wb') as f:
        pickle.dump(inverted_tag_index, f)
    logger.info(f"Saved inverted tag index to {INVERTED_INDEX_FILE}")

    # 6b. Save One-Hot DataFrame (which now includes 'genre_label' and 'genre_name')
    logger.info("Converting all sparse columns to dense 'int' type for parquet compatibility...")
    
    # Find all columns that are still SparseDtype
    sparse_cols = [col for col in one_hot_df.columns if pd.api.types.is_sparse(one_hot_df[col])]
    
    # Create a mapping to convert them all to a simple 'int'
    dtype_mapping = {col: 'int' for col in sparse_cols}

    # .astype() will convert all sparse columns to dense 'int'
    one_hot_df = one_hot_df.astype(dtype_mapping)

    logger.info("All sparse columns are now dense. Saving to parquet...")
    one_hot_df.to_parquet(TAGS_ONEHOT_FILE)
    logger.info(f"Saved one-hot tag matrix (with 'genre_label' and 'genre_name') to {TAGS_ONEHOT_FILE}")

    logger.info("\n--- V4.0 Pipeline - Phase 1 (Part B): Tag Lookup Generation COMPLETE ---")
    logger.info(f"Your lookup files now have categorized tags (e.g., 'genre::rock').")
    logger.info(f"Crucially, {TAGS_ONEHOT_FILE} now contains the 'genre_name' column for V4 tokenization.")
    logger.info("You can now run Phase 1.B (generate_triplets_v4.py).")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)