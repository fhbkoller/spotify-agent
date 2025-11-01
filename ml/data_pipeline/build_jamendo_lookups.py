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
    print("--- Phase 1 (Part B): Generating Tag Lookups ---")
    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load features.parquet to get all valid track_ids ---
    print(f"Loading features from {FEATURES_FILE} to get track IDs...")
    try:
        features_df = pd.read_parquet(FEATURES_FILE)
    except FileNotFoundError:
        print(f"FATAL ERROR: {FEATURES_FILE} not found.")
        print("Please run 'extract_features_pipeline_concurrent.py' (Phase 1A) first.")
        sys.exit(1)
        
    all_tracks_in_features = set(features_df.index.astype(str))
    print(f"Found {len(all_tracks_in_features)} processed tracks.")
    if not all_tracks_in_features:
        print("FATAL ERROR: No tracks found in features.parquet.")
        sys.exit(1)

    # --- 2. Download and Load Tag Data directly ---
    print("Starting Phase 1b: Downloading tag lookups from Hugging Face URLs...")
    try:
        print(f"Downloading {TRAIN_TSV_URL}...")
        df_train = pd.read_csv(TRAIN_TSV_URL, sep="\t")
        
        print(f"Downloading {VALID_TSV_URL}...")
        df_valid = pd.read_csv(VALID_TSV_URL, sep="\t")
        
        df_all_tags = pd.concat([df_train, df_valid], ignore_index=True)
        print(f"Successfully downloaded and combined {len(df_all_tags)} total metadata records.")

    except Exception as e:
        print(f"ERROR: Failed to download or parse .tsv files: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 3. Process Tags ---
    print("Parsing tag strings (this may take a minute)...")
    
    df_all_tags["genres_list"] = df_all_tags["genres"].apply(safe_eval)
    df_all_tags["instruments_list"] = df_all_tags["instruments"].apply(safe_eval)
    df_all_tags["moods_list"] = df_all_tags["moods"].apply(safe_eval)

    print("Cross-referencing tags with processed audio tracks...")
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
        print("ERROR: No matching tracks found between features.parquet and tag files.")
        sys.exit(1)
        
    print(f"Found {len(track_to_tags)} matching tracks with {len(all_tags)} unique *categorized* tags.")
    
    # --- 4. Build One-Hot Sparse Matrix ---
    print("Building sparse one-hot tag matrix...")
    
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

    # --- 5. NEW V3.0 STEP: Create the single-label 'genre_label' column ---
    print("Creating single-label 'genre_label' for V3 model...")
    
    # 5a. Get all genre columns, sorted alphabetically
    genre_cols = sorted([col for col in one_hot_df.columns if col.startswith('genre::')])
    
    if not genre_cols:
        print("ERROR: No 'genre::' columns found. Cannot create 'genre_label'.")
        sys.exit(1)
        
    print(f"Found {len(genre_cols)} unique genres. This will be our NUM_GENRES.")
    print(f"Example: '{genre_cols[0]}' -> 0, '{genre_cols[1]}' -> 1, etc.")
    
    # 5b. Get the subset of the DataFrame with only genre columns
    genre_df = one_hot_df[genre_cols]
    
    # 5c. Use argmax to find the integer index of the first '1' in each row.
    # This effectively picks *one* genre for each track.
    genre_indices = genre_df.to_numpy().argmax(axis=1)
    
    # 5d. Check for tracks that have NO genre at all (sum of 1s is 0)
    has_no_genre = (genre_df.sum(axis=1) == 0)
    
    # 5e. Create the new column in the main DataFrame
    # Default to the argmax index
    one_hot_df['genre_label'] = genre_indices
    # Set tracks with no genre to -1 (we can filter these out later if needed)
    one_hot_df.loc[has_no_genre, 'genre_label'] = -1
    
    print("Successfully created 'genre_label' column.")

    # --- 6. Save Artifacts ---
    print("Saving lookup artifacts...")
    
    # 6a. Save Inverted Tag Index
    with open(INVERTED_INDEX_FILE, 'wb') as f:
        pickle.dump(inverted_tag_index, f)
    print(f"Saved inverted tag index to {INVERTED_INDEX_FILE}")

    # 6b. Save One-Hot DataFrame (which now includes 'genre_label')
    print("Converting all sparse columns to dense 'int' type for parquet compatibility...")
    
    # --- V3.0 FIX: ---
    # Find all columns that are still SparseDtype
    sparse_cols = [col for col in one_hot_df.columns if pd.api.types.is_sparse(one_hot_df[col])]
    
    # Create a mapping to convert them all to a simple 'int'
    dtype_mapping = {col: 'int' for col in sparse_cols}

    # .astype() will convert all sparse columns to dense 'int'
    # This is the correct way to de-sparsify for saving.
    one_hot_df = one_hot_df.astype(dtype_mapping)

    print("All columns are now dense. Saving to parquet...")
    one_hot_df.to_parquet(TAGS_ONEHOT_FILE)
    print(f"Saved one-hot tag matrix (with 'genre_label') to {TAGS_ONEHOT_FILE}")

    print("\n--- Phase 1 (Part B): Tag Lookup Generation COMPLETE ---")
    print(f"Your lookup files now have categorized tags (e.g., 'genre::rock').")
    print(f"Crucially, {TAGS_ONEHOT_FILE} now contains the 'genre_label' column.")
    print("You can now run Phase 3 (generate_triplets.py).")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)