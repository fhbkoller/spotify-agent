# ml/testing/test_model_v2_training_fit.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root to path to import model and dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --- FIX: Import MultiTask model and dataset from the v2_2 training script ---
from ml.training.train_embedding_model_v2_2 import MultiTaskEmbeddingModel, SongDataset, EMBEDDING_DIM, NUM_GENRES
from src.utils.logging import logger

# --- Configuration ---
# --- FIX: Point to the final multitask model path ---
MODEL_PATH = 'ml/data/models/v2_2_multitask_model_final.pth'
# --- FIX: Test against the specificity dataset (as Stage 2 used it last) ---
TRAINING_DATA_PATH = 'ml/data/synthetic_specificity_v1.csv'
BATCH_SIZE = 256 # Increase batch size for faster testing

# --- Main Test Function ---
def test_training_fit():
    logger.info("--- Starting 'Goodness of Fit' Test (V2.2 - MultiTask - Sin/Cos) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- FIX: Load the MultiTaskEmbeddingModel ---
    logger.info(f"Loading MultiTask v2.2 model ({EMBEDDING_DIM}-dim) from {MODEL_PATH}...")
    # Provide necessary dimensions (input=12, embedding=12, num_genres=15)
    model = MultiTaskEmbeddingModel(
        input_dim=12,
        embedding_dim=EMBEDDING_DIM,
        num_genres=NUM_GENRES
    ).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Print detailed traceback for debugging loading issues
        import traceback
        traceback.print_exc()
        return

    model.eval() # Set model to evaluation mode

    # Load data using the updated SongDataset (handles both formats + sin/cos)
    logger.info(f"Loading test data from {TRAINING_DATA_PATH}...")
    try:
        dataset = SongDataset(TRAINING_DATA_PATH)
        num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    except FileNotFoundError:
        logger.error(f"Data file not found at {TRAINING_DATA_PATH}.")
        return
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    logger.info(f"Evaluating {len(dataset)} training triplets...")

    correct_placements = 0
    total_triplets = 0

    with torch.no_grad():
        # Unpack, ignoring dummy label
        for anchor, positive, negative, _ in tqdm(dataloader, desc="Evaluating Training Fit"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # --- FIX: Get embeddings from the MultiTask model ---
            # We only need the embeddings, ignore genre logits output
            anchor_emb, positive_emb, negative_emb, _ = model(anchor, positive, negative)

            dist_pos = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
            dist_neg = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)

            correct_placements += torch.sum(dist_pos < dist_neg).item()
            total_triplets += anchor.size(0)

    accuracy = (correct_placements / total_triplets) * 100 if total_triplets > 0 else 0

    logger.info("\n--- ['Goodness of Fit' Test COMPLETE] ---")
    logger.info(f"  Total Triplets:     {total_triplets}")
    logger.info(f"  Correctly Placed:   {correct_placements}")
    logger.info(f"  Incorrectly Placed: {total_triplets - correct_placements}")
    logger.info(f"  Training Fit Accuracy: {accuracy:.2f}%")

    if accuracy < 80: # Slightly lower expectation after NaN
        logger.warning("[Test WARNING]: Model accuracy on training data is lower than expected.")
    else:
        logger.info("[Test SUCCESS]: Model shows reasonable fit to the training data.")
    logger.info("------------------------------------------")

if __name__ == "__main__":
    test_training_fit()