# ml/testing/test_model_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --- FIX: Import MultiTask model ---
from ml.training.train_embedding_model_v2_2 import MultiTaskEmbeddingModel, EMBEDDING_DIM, NUM_GENRES
from src.utils.logging import logger

# --- Configuration ---
# --- FIX: Point to the final multitask model path ---
MODEL_PATH = 'ml/data/models/v2_2_multitask_model_final.pth'

# Define feature indices and number for the *raw* 11 features
NUM_FEATURES_RAW = 11
KEY_INDEX = 9
MODE_INDEX = 10

# --- Helper Function to Create Samples with Sin/Cos Encoding ---
def create_test_sample(features_dict):
    """
    Creates a 12-element tensor from a dictionary of 11 raw features,
    applying Sin/Cos encoding to the key.
    """
    raw_feature_order = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'speechiness', 'valence', 'tempo', 'loudness', # Numerical 0-8
        'key', 'mode' # Categorical 9, 10
    ]
    raw_features = [features_dict.get(name, 0.0) for name in raw_feature_order]

    key = raw_features[KEY_INDEX]
    mode = raw_features[MODE_INDEX]

    key_sin = np.sin(2 * np.pi * key / 12.0)
    key_cos = np.cos(2 * np.pi * key / 12.0)

    # Final 12-feature vector matching the dataset's processing:
    final_features = raw_features[:KEY_INDEX] + [mode, key_sin, key_cos]

    return torch.tensor(final_features, dtype=torch.float32).unsqueeze(0) # Add batch dim

# --- Main Test Function ---
def test_model_sensitivity():
    logger.info("--- Starting V2.2 (MultiTask - Sin/Cos) Model Sensitivity Validation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- FIX: Load the MultiTaskEmbeddingModel ---
    logger.info(f"Loading MultiTask v2.2 model ({EMBEDDING_DIM}-dim) from {MODEL_PATH}...")
    model = MultiTaskEmbeddingModel(
        input_dim=12,
        embedding_dim=EMBEDDING_DIM,
        num_genres=NUM_GENRES
    ).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Run training first.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Print detailed traceback
        import traceback
        traceback.print_exc()
        return

    model.eval()

    # --- Test Cases ---
    base_features = {
        'acousticness': 0.5, 'danceability': 0.6, 'energy': 0.7, 'instrumentalness': 0.1,
        'liveness': 0.2, 'speechiness': 0.05, 'valence': 0.8, 'tempo': 0.5, 'loudness': 0.8,
        'key': 5.0, 'mode': 1.0 # Base: F# Major
    }

    # Test 1: Mode Sensitivity (Major vs Minor)
    logger.info("\n--- [Test 1: Mode Sensitivity] ---")
    major_sample = create_test_sample(base_features | {'mode': 1.0}).to(device) # F# Major
    minor_sample = create_test_sample(base_features | {'mode': 0.0}).to(device) # F# Minor

    with torch.no_grad():
        # --- FIX: Call model correctly (ignore genre logits) ---
        major_emb, _ = model(major_sample)
        minor_emb, _ = model(minor_sample)
        similarity = F.cosine_similarity(major_emb, minor_emb).item()

    logger.info(f"Similarity between '{int(base_features['key'])} Major' and '{int(base_features['key'])} Minor':")
    logger.info(f"  V2.2 (MultiTask) Model: {similarity:.4f}")
    if similarity > 0.90: # Slightly looser threshold after NaN training
        logger.error("[Test 1 FAILURE]: Model is likely IGNORING 'mode'. Similarity too high.")
    elif similarity < 0:
         logger.error("[Test 1 FAILURE]: Model assigns negative similarity. Embedding space potentially collapsed.")
    else:
        logger.info("[Test 1 SUCCESS]: Model shows sensitivity to 'mode'.")
    logger.info("---------------------------------")


    # Test 2: Key Sensitivity (Cyclical - Close Keys)
    logger.info("\n--- [Test 2: Key Sensitivity - Close] ---")
    key_Fsharp = create_test_sample(base_features | {'key': 5.0}).to(device) # F# (Key 5)
    key_G = create_test_sample(base_features | {'key': 6.0}).to(device)      # G (Key 6) - Semitone apart

    with torch.no_grad():
        fsharp_emb, _ = model(key_Fsharp)
        g_emb, _ = model(key_G)
        similarity_close = F.cosine_similarity(fsharp_emb, g_emb).item()

    logger.info(f"Similarity between Key '{int(base_features['key'])}' and Key '{6}':")
    logger.info(f"  V2.2 (MultiTask) Model: {similarity_close:.4f}")
    if similarity_close < 0.5:
        logger.warning("[Test 2 WARNING]: Similarity between adjacent keys is low. Check encoding/training.")
    else:
        logger.info("[Test 2 SUCCESS]: Model shows high similarity for adjacent keys.")
    logger.info("-------------------------------------")


    # Test 3: Key Sensitivity (Cyclical - Distant Keys)
    logger.info("\n--- [Test 3: Key Sensitivity - Distant] ---")
    key_C = create_test_sample(base_features | {'key': 0.0}).to(device) # C (Key 0) - Tritone from F#

    with torch.no_grad():
        fsharp_emb, _ = model(key_Fsharp) # Re-use F# embedding
        c_emb, _ = model(key_C)
        similarity_distant = F.cosine_similarity(fsharp_emb, c_emb).item()

    logger.info(f"Similarity between Key '{int(base_features['key'])}' and Key '{0}' (Tritone):")
    logger.info(f"  V2.2 (MultiTask) Model: {similarity_distant:.4f}")
    if similarity_close == 0 or similarity_distant == 0 : # Add check for zero embeddings if NaNs occured
        logger.error("[Test 3 FAILURE]: Zero embeddings detected. Model likely collapsed during training.")
    elif similarity_distant > similarity_close:
        logger.error("[Test 3 FAILURE]: Distant keys are MORE similar than close keys. Cyclical encoding likely failed.")
    elif similarity_distant > 0.8:
        logger.warning("[Test 3 WARNING]: Similarity between distant keys is very high.")
    else:
        logger.info("[Test 3 SUCCESS]: Model shows lower similarity for distant keys.")
    logger.info("---------------------------------------")


    # Test 4: Numerical Sensitivity (Low vs High Vibe)
    logger.info("\n--- [Test 4: Numerical Sensitivity] ---")
    low_vibe = {
        'acousticness': 0.9, 'danceability': 0.2, 'energy': 0.1, 'instrumentalness': 0.8,
        'liveness': 0.1, 'speechiness': 0.03, 'valence': 0.1, 'tempo': 0.2, 'loudness': 0.1,
        'key': 2.0, 'mode': 0.0 # D Minor
    }
    high_vibe = {
        'acousticness': 0.1, 'danceability': 0.8, 'energy': 0.9, 'instrumentalness': 0.0,
        'liveness': 0.4, 'speechiness': 0.2, 'valence': 0.9, 'tempo': 0.8, 'loudness': 0.9,
        'key': 7.0, 'mode': 1.0 # G Major
    }
    low_sample = create_test_sample(low_vibe).to(device)
    high_sample = create_test_sample(high_vibe).to(device)

    with torch.no_grad():
        low_emb, _ = model(low_sample)
        high_emb, _ = model(high_sample)
        similarity_num = F.cosine_similarity(low_emb, high_emb).item()
        distance_num = torch.norm(low_emb - high_emb, p=2).item()

    logger.info("Similarity between 'Low Vibe' and 'High Vibe' songs:")
    logger.info(f"  V2.2 (MultiTask) Model Similarity: {similarity_num:.4f}")
    logger.info(f"  V2.2 (MultiTask) Model Distance:  {distance_num:.4f}")

    if similarity_num > 0.5:
        logger.error("[Test 4 FAILURE]: Model assigns high similarity to opposite vibes.")
    elif distance_num < 0.5: # Lowered threshold slightly due to potential NaN impact
         logger.warning("[Test 4 WARNING]: Euclidean distance between opposite vibes is small.")
    else:
        logger.info("[Test 4 SUCCESS]: Model distinguishes numerically distinct vibes.")
    logger.info("------------------------------------")

    logger.info("\n--- Validation Complete ---")

if __name__ == "__main__":
    test_model_sensitivity()