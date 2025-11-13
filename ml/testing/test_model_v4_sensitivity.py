# ml/testing/test_model_v4_sensitivity.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pandas as pd
from typing import Dict, Tuple

# Add project root for imports
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    src_path = os.path.join(project_root, 'src')
    if project_root not in sys.path: sys.path.insert(0, project_root)
    if src_path not in sys.path: sys.path.insert(1, src_path)
except NameError:
    project_root = os.path.abspath('.')
    src_path = os.path.join(project_root, 'src')
    if project_root not in sys.path: sys.path.insert(0, project_root)
    if src_path not in sys.path: sys.path.insert(1, src_path)

from src.utils.logging import logger

# --- V4.0 Imports ---
from ml.training.v4.architecture import (
    VibeFTTransformer, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, 
    OUTPUT_EMBEDDING_DIM, DROPOUT,
    NUMERICAL_FEATURE_KEYS_ORDERED,
    CATEGORICAL_FEATURE_KEYS_ORDERED
)

# --- Configuration ---
MODEL_PATH = 'ml/data/models/v4.0_transformer_model_final.pth'
TAGS_LOOKUP_PATH = 'ml/data/lookups/tags_onehot.parquet'

# --- V4.0 Helper: Build Genre Map (same as in training) ---
def build_genre_map(tags_parquet_path: str) -> Dict[str, int]:
    # This is a minimal version for the test script
    try:
        abs_tags_path = os.path.join(project_root, tags_parquet_path)
        tags_df = pd.read_parquet(abs_tags_path)
        genre_cols = sorted([col for col in tags_df.columns if col.startswith('genre::')])
        genre_map = {'genre::unknown': 0}
        for i, name in enumerate(genre_cols):
            genre_map[name] = i + 1
        # Add a fallback for the test
        genre_map['genre::rock'] = genre_map.get('genre::rock', 1) 
        return genre_map
    except Exception as e:
        logger.warning(f"Could not load genre map from {tags_parquet_path}: {e}. Using dummy map.")
        return {'genre::unknown': 0, 'genre::rock': 1}

# --- V4.0 Helper Function ---
def create_test_sample_v4(features_dict: Dict, genre_map: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates the (numerical, categorical, mask) tensors for the V4 model
    from a simple dictionary of raw feature names.
    """
    
    # 1. Numerical Tensor (B, 9)
    numerical_data = [features_dict.get(key, 0.5) for key in NUMERICAL_FEATURE_KEYS_ORDERED]
    num_tensor = torch.tensor(numerical_data, dtype=torch.float32).unsqueeze(0)
    
    # 2. Categorical Tensor (B, 3)
    genre_str = features_dict.get('genre', 'genre::rock')
    genre_idx = genre_map.get(genre_str, genre_map.get('genre::unknown', 0))
    key_val = features_dict.get('key', 0.0)
    mode_val = features_dict.get('mode', 0.0)
    
    cat_tensor = torch.tensor([genre_idx, key_val, mode_val], dtype=torch.long).unsqueeze(0)
    
    # 3. Mask Tensor (B, 12)
    # Assume all features are present for sensitivity testing
    mask_tensor = torch.full((1, 12), True, dtype=torch.bool)
    
    return num_tensor, cat_tensor, mask_tensor

# --- Main Test Function ---
def test_model_v4_sensitivity():
    logger.info("--- Starting V4.0 (FT-Transformer) Model Sensitivity Validation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- V4.0: Build Genre Map ---
    genre_to_index_map = build_genre_map(TAGS_LOOKUP_PATH)

    # --- V4.0: Load the VibeFTTransformer ---
    logger.info(f"Loading FT-Transformer v4.0 model ({OUTPUT_EMBEDDING_DIM}-dim) from {MODEL_PATH}...")
    model = VibeFTTransformer(
        genre_to_index_map=genre_to_index_map,
        d_model=D_MODEL,
        n_head=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        output_dim=OUTPUT_EMBEDDING_DIM,
        dropout=DROPOUT
    ).to(device)
    
    if device.type == 'cuda':
        logger.info("Applying torch.compile() for JIT optimization...")
        model = torch.compile(model)
        
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Run training first.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    model.eval()

    # --- Test Cases ---
    base_features = {
        'acousticness': 0.5, 'danceability': 0.6, 'energy': 0.7, 'instrumentalness': 0.1,
        'liveness': 0.2, 'speechiness': 0.05, 'valence': 0.8, 'tempo': 0.5, 'loudness': 0.8,
        'key': 6.0, 'mode': 1.0, 'genre': 'genre::rock' # Base: G Major Rock
    }
    
    # Helper to get embedding
    def get_emb(features: Dict) -> torch.Tensor:
        num, cat, mask = create_test_sample_v4(features, genre_to_index_map)
        num, cat, mask = num.to(device), cat.to(device), mask.to(device)
        with torch.no_grad():
            emb, _, _ = model(num, cat, mask, apply_feature_dropout=False)
        return emb

    # Test 1: Mode Sensitivity (Major vs Minor)
    logger.info("\n--- [Test 1: Mode Sensitivity] ---")
    major_emb = get_emb(base_features | {'mode': 1.0}) # G Major
    minor_emb = get_emb(base_features | {'mode': 0.0}) # G Minor
    similarity = F.cosine_similarity(major_emb, minor_emb).item()

    logger.info(f"Similarity between '{int(base_features['key'])} Major' and '{int(base_features['key'])} Minor':")
    logger.info(f"  V4.0 (Transformer) Model: {similarity:.4f}")
    
    if similarity > 0.5:
        logger.error("[Test 1 FAILURE]: Model is IGNORING 'mode'. Similarity too high.")
    elif similarity < -0.5:
         logger.info(f"[Test 1 SUCCESS]: Model shows strong NEGATIVE similarity to 'mode' (Similarity: {similarity:.4f}). This is excellent.")
    else:
        logger.info(f"[Test 1 SUCCESS]: Model shows strong sensitivity to 'mode' (Similarity: {similarity:.4f}).")
    logger.info("---------------------------------")


    # Test 2: Key Sensitivity (Cyclical - Close Keys: C -> B)
    logger.info("\n--- [Test 2: Key Sensitivity - Close (Semitone)] ---")
    base_key_C = base_features | {'key': 0.0} # Base C Major
    
    c_emb = get_emb(base_key_C)         # C (Key 0)
    b_emb = get_emb(base_key_C | {'key': 11.0}) # B (Key 11) - Semitone
    similarity_close = F.cosine_similarity(c_emb, b_emb).item()

    logger.info(f"Similarity between Key '0' (C) and Key '11' (B) [Close]:")
    logger.info(f"  V4.0 (Transformer) Model: {similarity_close:.4f}")
    if similarity_close < 0.7: # Expect high similarity
        logger.warning("[Test 2 WARNING]: Similarity between adjacent keys (C, B) is low. Cyclical encoding may be weak.")
    else:
        logger.info("[Test 2 SUCCESS]: Model shows high similarity for adjacent keys.")
    logger.info("-------------------------------------")


    # Test 3: Key Sensitivity (Cyclical - Distant Keys: C -> F#)
    logger.info("\n--- [Test 3: Key Sensitivity - Distant (Tritone)] ---")
    fs_emb = get_emb(base_key_C | {'key': 6.0})  # F# (Key 6) - Tritone
    similarity_distant = F.cosine_similarity(c_emb, fs_emb).item()
        
    logger.info(f"Similarity between Key '0' (C) and Key '6' (F#) [Distant]:")
    logger.info(f"  V4.0 (Transformer) Model: {similarity_distant:.4f}")

    if similarity_distant > similarity_close:
        logger.error("[Test 3 FAILURE]: Distant keys (C, F#) are MORE similar than close keys (C, B). Cyclical encoding FAILED.")
    elif similarity_distant > 0.5:
        logger.warning(f"[Test 3 WARNING]: Similarity between distant keys is high ({similarity_distant:.4f}).")
    else:
        logger.info(f"[Test 3 SUCCESS]: Model shows lower similarity for distant keys (Dist: {similarity_distant:.4f} < Close: {similarity_close:.4f}).")
    logger.info("---------------------------------------")


    # Test 4: Numerical Sensitivity (Low vs High Vibe)
    logger.info("\n--- [Test 4: Numerical Sensitivity] ---")
    low_vibe = {
        'acousticness': 0.9, 'danceability': 0.2, 'energy': 0.1, 'instrumentalness': 0.8,
        'liveness': 0.1, 'speechiness': 0.03, 'valence': 0.1, 'tempo': 0.2, 'loudness': 0.1,
        'key': 2.0, 'mode': 0.0, 'genre': 'genre::ambient' # D Minor Ambient
    }
    high_vibe = {
        'acousticness': 0.1, 'danceability': 0.8, 'energy': 0.9, 'instrumentalness': 0.0,
        'liveness': 0.4, 'speechiness': 0.2, 'valence': 0.9, 'tempo': 0.8, 'loudness': 0.9,
        'key': 7.0, 'mode': 1.0, 'genre': 'genre::dancepop' # G Major Dancepop
    }
    low_emb = get_emb(low_vibe)
    high_emb = get_emb(high_vibe)
    similarity_num = F.cosine_similarity(low_emb, high_emb).item()
    distance_num = torch.norm(low_emb - high_emb, p=2).item()

    logger.info("Similarity between 'Low Vibe' (Ambient) and 'High Vibe' (Dancepop) songs:")
    logger.info(f"  V4.0 (Transformer) Model Similarity: {similarity_num:.4f}")
    logger.info(f"  V4.0 (Transformer) Model Distance:  {distance_num:.4f}")

    if similarity_num > 0.1: # Expect very low similarity
        logger.error("[Test 4 FAILURE]: Model assigns high similarity to opposite vibes.")
    elif distance_num < 1.0: # Expect large distance
         logger.warning("[Test 4 WARNING]: Euclidean distance between opposite vibes is small.")
    else:
        logger.info("[Test 4 SUCCESS]: Model strongly distinguishes numerically distinct vibes.")
    logger.info("------------------------------------")

    logger.info("\n--- V4.0 Validation Complete ---")

if __name__ == "__main__":
    test_model_v4_sensitivity()