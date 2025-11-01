# ml/testing/test_model_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --- V3.0: Import V3 model and config from the new training script ---
from ml.training.train_embedding_model_v3 import MultiTaskEmbeddingModel_V3, EMBEDDING_DIM, NUM_GENRES, NUM_KEYS
from src.utils.logging import logger

# --- Configuration ---
# --- V3.0: Point to the final V3 model path ---
MODEL_PATH = 'ml/data/models/v3.0_multitask_model_final.pth'

# Define feature indices and number for the *raw* 11 features
KEY_INDEX = 9
MODE_INDEX = 10

# --- Helper Function (Unchanged) ---
def create_test_sample(features_dict):
    """
    Creates a 12-element tensor from a dictionary of 11 raw features,
    applying Sin/Cos encoding to the key. (Matches training preprocessing)
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
def test_model_v3_sensitivity():
    logger.info("--- Starting V3.0 (MultiTask) Model Sensitivity Validation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- V3.0: Load the MultiTaskEmbeddingModel_V3 ---
    logger.info(f"Loading V3.0 model ({EMBEDDING_DIM}-dim) from {MODEL_PATH}...")
    model = MultiTaskEmbeddingModel_V3(
        input_dim=12,
        embedding_dim=EMBEDDING_DIM,
        num_genres=NUM_GENRES, # Must match the 252 from training
        num_keys=NUM_KEYS
    ).to(device)
    
    # --- V3.0: Compile the model just like in training for max speed ---
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
        'key': 6.0, 'mode': 1.0 # Base: G Major (Key 6)
    }

    # Test 1: Mode Sensitivity (Major vs Minor)
    logger.info("\n--- [Test 1: Mode Sensitivity] ---")
    major_sample = create_test_sample(base_features | {'mode': 1.0}).to(device) # G Major
    minor_sample = create_test_sample(base_features | {'mode': 0.0}).to(device) # G Minor

    with torch.no_grad():
        # --- V3.0 FIX: Call new forward pass, take first output (embedding) ---
        major_emb, _, _, _ = model(major_sample)
        minor_emb, _, _, _ = model(minor_sample)
        similarity = F.cosine_similarity(major_emb, minor_emb).item()

    logger.info(f"Similarity between '{int(base_features['key'])} Major' and '{int(base_features['key'])} Minor':")
    logger.info(f"  V3.0 (MultiTask) Model: {similarity:.4f}")
    
    # We expect this to be very low, even negative.
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
    
    key_C_sample = create_test_sample(base_key_C).to(device)         # C (Key 0)
    key_B_sample = create_test_sample(base_key_C | {'key': 11.0}).to(device) # B (Key 11) - Semitone
    
    with torch.no_grad():
        # --- V3.0 FIX: Call new forward pass ---
        c_emb, _, _, _ = model(key_C_sample)
        b_emb, _, _, _ = model(key_B_sample)
        similarity_close = F.cosine_similarity(c_emb, b_emb).item()

    logger.info(f"Similarity between Key '0' (C) and Key '11' (B) [Close]:")
    logger.info(f"  V3.0 (MultiTask) Model: {similarity_close:.4f}")
    if similarity_close < 0.7: # Expect high similarity
        logger.warning("[Test 2 WARNING]: Similarity between adjacent keys (C, B) is low. Cyclical encoding may be weak.")
    else:
        logger.info("[Test 2 SUCCESS]: Model shows high similarity for adjacent keys.")
    logger.info("-------------------------------------")


    # Test 3: Key Sensitivity (Cyclical - Distant Keys: C -> F#)
    logger.info("\n--- [Test 3: Key Sensitivity - Distant (Tritone)] ---")
    key_Fs_sample = create_test_sample(base_key_C | {'key': 6.0}).to(device)  # F# (Key 6) - Tritone
    
    with torch.no_grad():
        # --- V3.0 FIX: Call new forward pass ---
        # c_emb is already computed
        fs_emb, _, _, _ = model(key_Fs_sample)
        similarity_distant = F.cosine_similarity(c_emb, fs_emb).item()
        
    logger.info(f"Similarity between Key '0' (C) and Key '6' (F#) [Distant]:")
    logger.info(f"  V3.0 (MultiTask) Model: {similarity_distant:.4f}")

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
        # --- V3.0 FIX: Call new forward pass ---
        low_emb, _, _, _ = model(low_sample)
        high_emb, _, _, _ = model(high_sample)
        similarity_num = F.cosine_similarity(low_emb, high_emb).item()
        distance_num = torch.norm(low_emb - high_emb, p=2).item()

    logger.info("Similarity between 'Low Vibe' and 'High Vibe' songs:")
    logger.info(f"  V3.0 (MultiTask) Model Similarity: {similarity_num:.4f}")
    logger.info(f"  V3.0 (MultiTask) Model Distance:  {distance_num:.4f}")

    if similarity_num > 0.1: # Expect very low similarity
        logger.error("[Test 4 FAILURE]: Model assigns high similarity to opposite vibes.")
    elif distance_num < 1.0: # Expect large distance
         logger.warning("[Test 4 WARNING]: Euclidean distance between opposite vibes is small.")
    else:
        logger.info("[Test 4 SUCCESS]: Model strongly distinguishes numerically distinct vibes.")
    logger.info("------------------------------------")

    logger.info("\n--- V3.0 Validation Complete ---")

if __name__ == "__main__":
    test_model_v3_sensitivity()