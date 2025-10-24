import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import logging
from tqdm import tqdm

# --- Configuration ---
V1_DATA_FILE = "data/training_data.csv"
V2_OUTPUT_FILE = "data/training_data_v2.csv" # Overwriting

# --- Mappings (Unchanged) ---
KEY_TO_INT_MAP = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
    'Unknown': -1
}
MODE_TO_INT_MAP = {
    'Minor': 0, 'Major': 1,
    'Unknown': -1
}

FEATURE_COLS = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'valence', 'tempo',
    'key', 'mode'
]
# *** NEW: Define indices for splitting
NUM_FEATURE_COUNT = 9
NUM_INDICES = list(range(NUM_FEATURE_COUNT))
CAT_INDICES = list(range(NUM_FEATURE_COUNT, len(FEATURE_COLS)))

PROMPT_REGEX = re.compile(r"([a-z]+)=([a-zA-Z0-9#\-]+)")

# --- Setup Logging (Unchanged) ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- parse_prompt_to_vector (Unchanged) ---
def parse_prompt_to_vector(prompt_str: str) -> list | None:
    try:
        parsed_features = dict(PROMPT_REGEX.findall(prompt_str))
        
        vector = []
        for feat_name in FEATURE_COLS:
            if feat_name not in parsed_features:
                logger.warning(f"Missing feature '{feat_name}' in prompt. Skipping.")
                return None
            
            val_str = parsed_features[feat_name]
            
            if feat_name == 'key':
                val_int = KEY_TO_INT_MAP.get(val_str, -1)
                if val_int == -1:
                    logger.warning(f"Unknown key: '{val_str}'. Skipping.")
                    return None
                vector.append(val_int)
            elif feat_name == 'mode':
                val_int = MODE_TO_INT_MAP.get(val_str, -1)
                if val_int == -1:
                    logger.warning(f"Unknown mode: '{val_str}'. Skipping.")
                    return None
                vector.append(val_int)
            else:
                vector.append(float(val_str))
        
        return vector
        
    except Exception as e:
        logger.error(f"Failed to parse prompt: {e}")
        return None

def main():
    logger.info(f"Loading v1 text data from {V1_DATA_FILE}...")
    try:
        df_v1 = pd.read_csv(V1_DATA_FILE)
        df_v1.dropna(inplace=True)
    except FileNotFoundError:
        logger.error(f"Error: {V1_DATA_FILE} not found.")
        return

    logger.info("Parsing all text prompts into numerical vectors...")
    
    tqdm.pandas(desc="Parsing Anchors")
    anchor_vecs = df_v1['anchor'].progress_apply(parse_prompt_to_vector)
    tqdm.pandas(desc="Parsing Positives")
    positive_vecs = df_v1['positive'].progress_apply(parse_prompt_to_vector)
    tqdm.pandas(desc="Parsing Negatives")
    negative_vecs = df_v1['negative'].progress_apply(parse_prompt_to_vector)

    df_v2 = pd.DataFrame({
        'anchor_vec': anchor_vecs,
        'positive_vec': positive_vecs,
        'negative_vec': negative_vecs
    })
    
    initial_count = len(df_v2)
    df_v2.dropna(inplace=True)
    final_count = len(df_v2)
    
    if final_count == 0:
        logger.error("No valid triplets were parsed.")
        return
    logger.info(f"Successfully parsed {final_count} triplets.")

    # --- 3. *** MODIFIED: Fit and Apply Scaler (Correctly) ---
    logger.info("Splitting numerical and categorical features...")
    
    # Stack all vectors into a giant [N, 11] array
    all_vectors = np.vstack([
        np.array(df_v2['anchor_vec'].tolist()),
        np.array(df_v2['positive_vec'].tolist()),
        np.array(df_v2['negative_vec'].tolist())
    ])
    
    # Split the data
    all_numerical_data = all_vectors[:, NUM_INDICES]
    all_categorical_data = all_vectors[:, CAT_INDICES]

    logger.info(f"Fitting MinMaxScaler *only* on {NUM_FEATURE_COUNT} numerical features...")
    scaler = MinMaxScaler()
    scaler.fit(all_numerical_data) # Fit *only* on numerical data

    logger.info("Scaling all numerical vectors...")
    all_numerical_scaled = scaler.transform(all_numerical_data)
    
    # Re-combine the scaled numerical data with the *original* categorical data
    all_vectors_corrected = np.hstack([
        all_numerical_scaled,
        all_categorical_data # Use the original, un-scaled integers
    ])
    
    # Un-stack the giant array back into anchors, positives, and negatives
    (anchors, positives, negatives) = np.vsplit(all_vectors_corrected, 3)

    # --- 4. Save to v2 CSV format ---
    logger.info(f"Saving {final_count} corrected triplets to {V2_OUTPUT_FILE}...")
    
    anchor_cols = [f'anchor_feat_{i}' for i in range(len(FEATURE_COLS))]
    positive_cols = [f'positive_feat_{i}' for i in range(len(FEATURE_COLS))]
    negative_cols = [f'negative_feat_{i}' for i in range(len(FEATURE_COLS))]

    df_anchors = pd.DataFrame(anchors, columns=anchor_cols)
    df_positives = pd.DataFrame(positives, columns=positive_cols)
    df_negatives = pd.DataFrame(negatives, columns=negative_cols)

    df_final = pd.concat([
        df_anchors.reset_index(drop=True), 
        df_positives.reset_index(drop=True), 
        df_negatives.reset_index(drop=True)
    ], axis=1)

    df_final.to_csv(V2_OUTPUT_FILE, index=False)
    logger.info("Conversion complete! CSV now contains 9 scaled floats and 2 raw integers per vector.")

if __name__ == "__main__":
    main()