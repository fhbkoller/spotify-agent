# ml/testing/test_model_v4_training_fit.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from typing import Dict

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
    OUTPUT_EMBEDDING_DIM, DROPOUT
)
from ml.training.v4.dataset import SongDatasetV4, v4_collate_fn_v2

# --- Configuration ---
MODEL_PATH = 'ml/data/models/v4.0_transformer_model_final.pth'
# Test against the fine-tuning data
TRAINING_DATA_PATH = 'ml/data/synthetic_specificity_v4.parquet'
TAGS_LOOKUP_PATH = 'ml/data/lookups/tags_onehot.parquet'
BATCH_SIZE = 1024 # Use a large batch size for fast eval

# --- V4.0 Helper: Build Genre Map (same as in training) ---
def build_genre_map(tags_parquet_path: str) -> Dict[str, int]:
    logger.info(f"Building genre_to_index_map from {tags_parquet_path}...")
    try:
        abs_tags_path = os.path.join(project_root, tags_parquet_path)
        tags_df = pd.read_parquet(abs_tags_path)
        genre_cols = sorted([col for col in tags_df.columns if col.startswith('genre::')])
        if not genre_cols:
            raise ValueError("No 'genre::' columns found in tags_onehot.parquet")
        
        genre_map = {'genre::unknown': 0}
        for i, name in enumerate(genre_cols):
            genre_map[name] = i + 1
        
        logger.info(f"Built map with {len(genre_map)} genres (index 0 = 'unknown').")
        return genre_map
        
    except FileNotFoundError:
        logger.error(f"FATAL: {abs_tags_path} not found. Cannot build genre map.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: Error building genre map: {e}")
        sys.exit(1)

# --- Main Test Function ---
def test_training_fit_v4():
    logger.info("--- Starting 'Goodness of Fit' Test (V4.0 - FT-Transformer) ---")
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
        logger.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    model.eval() # Set model to evaluation mode

    # --- V4.0: Load data using the V4 Dataset and Collate ---
    logger.info(f"Loading test data from {TRAINING_DATA_PATH}...")
    try:
        dataset = SongDatasetV4(TRAINING_DATA_PATH, genre_to_index_map)
        num_workers = min(8, os.cpu_count()) if os.cpu_count() else 0
        pin_memory = True
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            collate_fn=v4_collate_fn_v2
        )
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
        # --- V4.0: Unpack all 3 tuples (A, P, N), each containing (num, cat, mask) ---
        for anchor_tensors, positive_tensors, negative_tensors in tqdm(dataloader, desc="Evaluating Training Fit"):
            
            anchor_num, anchor_cat, anchor_mask = anchor_tensors
            pos_num, pos_cat, pos_mask = positive_tensors
            neg_num, neg_cat, neg_mask = negative_tensors

            # --- V4.0: Use async transfer ---
            anchor_num = anchor_num.to(device, non_blocking=True)
            anchor_cat = anchor_cat.to(device, non_blocking=True)
            anchor_mask = anchor_mask.to(device, non_blocking=True)
            pos_num = pos_num.to(device, non_blocking=True)
            pos_cat = pos_cat.to(device, non_blocking=True)
            pos_mask = pos_mask.to(device, non_blocking=True)
            neg_num = neg_num.to(device, non_blocking=True)
            neg_cat = neg_cat.to(device, non_blocking=True)
            neg_mask = neg_mask.to(device, non_blocking=True)

            # --- V4.0: Combine all inputs for a single forward pass ---
            all_num = torch.cat([anchor_num, pos_num, neg_num], dim=0)
            all_cat = torch.cat([anchor_cat, pos_cat, neg_cat], dim=0)
            all_mask = torch.cat([anchor_mask, pos_mask, neg_mask], dim=0)

            # Run forward pass, no dropout, get embedding only
            all_embs, _, _ = model(
                numerical_data=all_num,
                categorical_data=all_cat,
                feature_mask=all_mask,
                apply_feature_dropout=False
            )
            
            anchor_emb, positive_emb, negative_emb = torch.chunk(all_embs, 3, dim=0)

            dist_pos = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
            dist_neg = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)

            correct_placements += torch.sum(dist_pos < dist_neg).item()
            total_triplets += anchor_emb.size(0)

    accuracy = (correct_placements / total_triplets) * 100 if total_triplets > 0 else 0

    logger.info("\n--- ['Goodness of Fit' Test COMPLETE] ---")
    logger.info(f"  Total Triplets:     {total_triplets}")
    logger.info(f"  Correctly Placed:   {correct_placements}")
    logger.info(f"  Incorrectly Placed: {total_triplets - correct_placements}")
    logger.info(f"  Training Fit Accuracy: {accuracy:.2f}%")

    if accuracy < 95: # We expect very high accuracy on data it was fine-tuned on
        logger.warning("[Test WARNING]: Model accuracy on training data is lower than 95%.")
    else:
        logger.info("[Test SUCCESS]: Model shows excellent fit to the training data.")
    logger.info("------------------------------------------")

if __name__ == "__main__":
    test_training_fit_v4()