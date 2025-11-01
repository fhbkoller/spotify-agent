# ml/testing/test_model_v3_training_fit.py

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
# --- V3.0: Import V3 model and dataset ---
from ml.training.train_embedding_model_v3 import MultiTaskEmbeddingModel_V3, SongDatasetV3, EMBEDDING_DIM, NUM_GENRES, NUM_KEYS
from src.utils.logging import logger

# --- Configuration ---
# --- V3.0: Point to the final V3 model path ---
MODEL_PATH = 'ml/data/models/v3.0_multitask_model_final.pth'
# --- V3.0: Test against the specificity dataset (as Stage 2 used it last) ---
TRAINING_DATA_PATH = 'ml/data/synthetic_specificity_v2.csv'
BATCH_SIZE = 1024 # Use a large batch size for fast eval

# --- Main Test Function ---
def test_training_fit():
    logger.info("--- Starting 'Goodness of Fit' Test (V3.0 - MultiTask) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- V3.0: Load the MultiTaskEmbeddingModel_V3 ---
    logger.info(f"Loading MultiTask v3.0 model ({EMBEDDING_DIM}-dim) from {MODEL_PATH}...")
    model = MultiTaskEmbeddingModel_V3(
        input_dim=12,
        embedding_dim=EMBEDDING_DIM,
        num_genres=NUM_GENRES,
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
        logger.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    model.eval() # Set model to evaluation mode

    # --- V3.0: Load data using the V3 "Pre-Baked CPU" SongDataset ---
    logger.info(f"Loading test data from {TRAINING_DATA_PATH}...")
    try:
        # Use the "Producer-Consumer" setup (data in CPU RAM, workers load)
        dataset = SongDatasetV3(TRAINING_DATA_PATH)
        num_workers = min(8, os.cpu_count()) if os.cpu_count() else 0
        pin_memory = True
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
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
        # --- V3.0: Unpack all 6 items from the V3 dataset ---
        for anchor, positive, negative, _, _, _ in tqdm(dataloader, desc="Evaluating Training Fit"):
            # --- V3.0: Use async transfer ---
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            # --- V3.0: Use new model forward pass ---
            combined_input = torch.cat((anchor, positive, negative), dim=0)
            all_embs, _, _, _ = model(combined_input)
            anchor_emb, positive_emb, negative_emb = torch.chunk(all_embs, 3, dim=0)

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

    if accuracy < 90: # We expect high accuracy on data it was trained on
        logger.warning("[Test WARNING]: Model accuracy on training data is lower than expected.")
    else:
        logger.info("[Test SUCCESS]: Model shows excellent fit to the training data.")
    logger.info("------------------------------------------")

if __name__ == "__main__":
    test_training_fit()