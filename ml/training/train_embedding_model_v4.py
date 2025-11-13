# ml/training/train_embedding_model_v4.py
# (Refactored V5.1: Disables MFM loss in Stage 2 to prevent collapse)

import os
import sys

# --- V5.0 ROBUST PATHING FIX ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# --- END V5.0 FIX ---

# --- DATA DIR FIX ---
try:
    log_dir = os.path.join(project_root, "data")
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f"CRITICAL: Could not create data directory at {log_dir}: {e}")
# --- END DATA DIR FIX ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# --- V5.1: TF32 API FIX ---
# Set the new precision APIs to silence warnings
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
# --- END V5.1 FIX ---

from src.utils.logging import logger
from ml.training.v4.dataset import SongDatasetV4, v4_collate_fn_v2
from ml.training.v4.architecture import (
    VibeFTTransformer, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, 
    OUTPUT_EMBEDDING_DIM, DROPOUT
)

# --- Configuration ---
JAMENDO_DATASET_PATH = 'ml/data/triplets_v4.parquet'
SYNTHETIC_DATASET_PATH = 'ml/data/synthetic_specificity_v4.parquet'
TAGS_LOOKUP_PATH = 'ml/data/lookups/tags_onehot.parquet'
MODEL_SAVE_PATH = 'ml/data/models/v4.0_transformer_model_final.pth'

# --- Hyperparameters ---
BATCH_SIZE_STAGE1 = 2048
BATCH_SIZE_STAGE2 = 1024
EPOCHS_STAGE1 = 50  # We know it converges by 50
EPOCHS_STAGE2 = 100
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5

TRIPLET_MARGIN = 0.5
W_MFM_S1 = 0.5

# --- V5.1 FIX: TURN OFF MFM LOSS IN STAGE 2 ---
# This is critical. The MFM task was starving the Triplet task,
# causing the model to collapse and not learn embeddings.
W_MFM_S2 = 0.0
# --- END V5.1 FIX ---

GRAD_CLIP_NORM = 1.0

# --- Helper: Build Genre Map (Unchanged) ---
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

# --- Helper: Triplet Mining (Unchanged) ---
def online_semi_hard_triplet_mining(anchor_emb, positive_emb, negative_emb, margin):
    d_ap = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
    d_an = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)
    is_semi_hard = (d_an > d_ap) & (d_an < d_ap + margin)
    
    if not is_semi_hard.any():
        is_hard = d_an < d_ap # Fallback to HARD triplets
        if not is_hard.any():
            return None, None, None
        anchor_final = anchor_emb[is_hard]
        positive_final = positive_emb[is_hard]
        negative_final = negative_emb[is_hard]
        return anchor_final, positive_final, negative_final

    anchor_final = anchor_emb[is_semi_hard]
    positive_final = positive_emb[is_semi_hard]
    negative_final = negative_emb[is_semi_hard]
    return anchor_final, positive_final, negative_final

# --- MFM Loss Calculation (Unchanged) ---
mfm_loss_num_fn = nn.MSELoss(reduction='none')
mfm_loss_cat_fn = nn.CrossEntropyLoss(reduction='none')

def calculate_mfm_loss(
    mfm_predictions: Tuple,
    targets_num: torch.Tensor,
    targets_cat: torch.Tensor,
    dropout_mask: torch.Tensor
) -> torch.Tensor:
    
    (genre_logits, key_logits, mode_logits, numerical_preds) = mfm_predictions
    
    loss_genre = mfm_loss_cat_fn(genre_logits, targets_cat[:, 0])
    loss_key = mfm_loss_cat_fn(key_logits, targets_cat[:, 1])
    loss_mode = mfm_loss_cat_fn(mode_logits, targets_cat[:, 2])
    loss_num = mfm_loss_num_fn(numerical_preds, targets_num)
    
    loss_cat_all = torch.stack([loss_genre, loss_key, loss_mode], dim=1)
    loss_all = torch.cat([loss_cat_all, loss_num], dim=1)
    
    masked_loss_all = loss_all * dropout_mask.float()
    
    num_masked_elements = dropout_mask.sum()
    if num_masked_elements == 0:
        return torch.tensor(0.0, device=loss_all.device)
        
    total_mfm_loss = masked_loss_all.sum() / num_masked_elements
    
    return total_mfm_loss

# --- Training Loop ---
def train_v4_model():
    logger.info("--- Starting Training: V4.2 FT-Transformer (w/ MFM Loss & 128-dim) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == 'cpu':
        logger.warning("Running on CPU. Training will be very slow.")
        
    # --- 1. Build Genre Map ---
    genre_to_index_map = build_genre_map(TAGS_LOOKUP_PATH)
    
    # --- 2. Initialize Model ---
    logger.info("Initializing V4.2 Model...")
    
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
        
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel training.")
        model = nn.DataParallel(model)
        
    logger.info(f"Model configured with NUM_GENRES = {len(genre_to_index_map)} and OUTPUT_DIM = {OUTPUT_EMBEDDING_DIM}")
    
    # --- 3. Loss & Optimizer ---
    triplet_loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    logger.info(f"Initializing GradScaler (Enabled: {device.type == 'cuda'})...")

    # --- 4. Setup DataLoaders ---
    num_workers = min(8, os.cpu_count()) if os.cpu_count() else 0
    logger.info(f"Using {num_workers} data loader workers.")
    
    # --- Stage 1: Pre-training on Jamendo V4 ---
    logger.info("\n--- Stage 1: Pre-training on Jamendo (V4 Dataset) ---")
    logger.info(f"Loading Jamendo dataset from {JAMENDO_DATASET_PATH}...")
    try:
        abs_jamendo_path = os.path.join(project_root, JAMENDO_DATASET_PATH)
        jamendo_dataset = SongDatasetV4(abs_jamendo_path, genre_to_index_map)
        
        jamendo_dataloader = DataLoader(
            jamendo_dataset, 
            batch_size=BATCH_SIZE_STAGE1,
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=v4_collate_fn_v2, 
            pin_memory=True
        )
        logger.info(f"Loaded {len(jamendo_dataset)} Jamendo triplets.")
    except Exception as e:
        logger.error(f"Error loading Jamendo dataset: {e}", exc_info=True)
        return

    optimizer_stage1 = optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE1)
    logger.info(f"  Epochs: {EPOCHS_STAGE1} | LR: {LEARNING_RATE_STAGE1} | Batch: {BATCH_SIZE_STAGE1}")
    logger.info(f"  Loss: Triplet (Margin={TRIPLET_MARGIN}) + MFM (Weight={W_MFM_S1}, ENABLED)")
    logger.info("  Training with Feature Dropout (p=0.25)")
    
    model.train()
    for epoch in range(EPOCHS_STAGE1):
        total_loss_s1, total_triplet_s1, total_mfm_s1, num_batches_s1 = 0.0, 0.0, 0.0, 0
        
        pbar = tqdm(jamendo_dataloader, desc=f"Stage 1 - Epoch {epoch+1}/{EPOCHS_STAGE1}")
        
        for anchor_tensors, positive_tensors, negative_tensors in pbar:
            
            if anchor_tensors[0].shape[0] == 0: 
                continue
                
            optimizer_stage1.zero_grad()
            
            anchor_num_gpu = anchor_tensors[0].to(device, non_blocking=True)
            anchor_cat_gpu = anchor_tensors[1].to(device, non_blocking=True)
            anchor_mask_gpu = anchor_tensors[2].to(device, non_blocking=True)
            
            pos_num_gpu = positive_tensors[0].to(device, non_blocking=True)
            pos_cat_gpu = positive_tensors[1].to(device, non_blocking=True)
            pos_mask_gpu = positive_tensors[2].to(device, non_blocking=True)
            
            neg_num_gpu = negative_tensors[0].to(device, non_blocking=True)
            neg_cat_gpu = negative_tensors[1].to(device, non_blocking=True)
            neg_mask_gpu = negative_tensors[2].to(device, non_blocking=True)

            all_num = torch.cat([anchor_num_gpu, pos_num_gpu, neg_num_gpu], dim=0)
            all_cat = torch.cat([anchor_cat_gpu, pos_cat_gpu, neg_cat_gpu], dim=0)
            all_mask = torch.cat([anchor_mask_gpu, pos_mask_gpu, neg_mask_gpu], dim=0)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                all_embeddings, mfm_predictions, mfm_dropout_mask = model(
                    numerical_data=all_num,
                    categorical_data=all_cat,
                    feature_mask=all_mask,
                    apply_feature_dropout=True, 
                    dropout_rate=0.25
                )
                
                (anchor_emb, positive_emb, negative_emb) = torch.chunk(all_embeddings, 3, dim=0)

                (anchor_final, positive_final, negative_final) = online_semi_hard_triplet_mining(
                    anchor_emb, positive_emb, negative_emb, TRIPLET_MARGIN
                )
                
                loss_triplet = torch.tensor(0.0, device=device)
                if anchor_final is not None and anchor_final.shape[0] > 0:
                    loss_triplet = triplet_loss_fn(anchor_final, positive_final, negative_final)

                loss_mfm = torch.tensor(0.0, device=device)
                if mfm_predictions is not None:
                    loss_mfm = calculate_mfm_loss(
                        mfm_predictions,
                        targets_num=all_num,
                        targets_cat=all_cat,
                        dropout_mask=mfm_dropout_mask
                    )
                
                total_batch_loss = loss_triplet + (W_MFM_S1 * loss_mfm)

            if torch.isnan(total_batch_loss):
                logger.error(f"NaN loss detected at Stage 1, Epoch {epoch+1}. Stopping training.")
                return

            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer_stage1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer_stage1)
            scaler.update()

            total_loss_s1 += total_batch_loss.item()
            total_triplet_s1 += loss_triplet.item()
            total_mfm_s1 += loss_mfm.item() 
            num_batches_s1 += 1
            if num_batches_s1 > 0:
                pbar.set_postfix(
                    Loss=f"{(total_loss_s1/num_batches_s1):.4f}", 
                    Triplet=f"{(total_triplet_s1/num_batches_s1):.4f}",
                    MFM=f"{(total_mfm_s1/num_batches_s1):.4f}"
                )
            
    logger.info("--- Stage 1 Complete ---")

    # --- Stage 2: Fine-tuning on Specificity V4 ---
    logger.info("\n--- Stage 2: Fine-tuning on Specificity (V4 Dataset) ---")
    logger.info(f"Loading Specificity dataset from {SYNTHETIC_DATASET_PATH}...")
    try:
        abs_synth_path = os.path.join(project_root, SYNTHETIC_DATASET_PATH)
        spec_dataset = SongDatasetV4(abs_synth_path, genre_to_index_map)
        
        spec_dataloader = DataLoader(
            spec_dataset, 
            batch_size=BATCH_SIZE_STAGE2,
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=v4_collate_fn_v2, 
            pin_memory=True
        )
        logger.info(f"Loaded {len(spec_dataset)} Specificity triplets.")
    except Exception as e:
        logger.error(f"Error loading Specificity dataset: {e}", exc_info=True)
        return

    optimizer_stage2 = optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE2)
    
    # --- V5.1 FIX: Update log message ---
    logger.info(f"  Epochs: {EPOCHS_STAGE2} | LR: {LEARNING_RATE_STAGE2} | Batch: {BATCH_SIZE_STAGE2}")
    logger.info(f"  Loss: Triplet (Margin={TRIPLET_MARGIN}) + MFM (Weight={W_MFM_S2}, DISABLED)")
    logger.info("  Training with Feature Dropout (p=0.25) for MFM (but MFM loss is 0)")
    # --- END V5.1 FIX ---

    model.train()
    for epoch in range(EPOCHS_STAGE2):
        total_loss_s2, total_triplet_s2, total_mfm_s2, num_batches_s2 = 0.0, 0.0, 0.0, 0
        
        pbar = tqdm(spec_dataloader, desc=f"Stage 2 - Epoch {epoch+1}/{EPOCHS_STAGE2}")
        for anchor_tensors, positive_tensors, negative_tensors in pbar:
            
            if anchor_tensors[0].shape[0] == 0: continue
                
            optimizer_stage2.zero_grad()
            
            anchor_num_gpu = anchor_tensors[0].to(device, non_blocking=True)
            anchor_cat_gpu = anchor_tensors[1].to(device, non_blocking=True)
            anchor_mask_gpu = anchor_tensors[2].to(device, non_blocking=True)
            
            pos_num_gpu = positive_tensors[0].to(device, non_blocking=True)
            pos_cat_gpu = positive_tensors[1].to(device, non_blocking=True)
            pos_mask_gpu = positive_tensors[2].to(device, non_blocking=True)
            
            neg_num_gpu = negative_tensors[0].to(device, non_blocking=True)
            neg_cat_gpu = negative_tensors[1].to(device, non_blocking=True)
            neg_mask_gpu = negative_tensors[2].to(device, non_blocking=True)

            all_num = torch.cat([anchor_num_gpu, pos_num_gpu, neg_num_gpu], dim=0)
            all_cat = torch.cat([anchor_cat_gpu, pos_cat_gpu, neg_cat_gpu], dim=0)
            all_mask = torch.cat([anchor_mask_gpu, pos_mask_gpu, neg_mask_gpu], dim=0)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                
                # --- V5.1 FIX: Don't apply feature dropout if MFM is off ---
                # We want the model to see all features for the triplet task
                all_embeddings, mfm_predictions, mfm_dropout_mask = model(
                    numerical_data=all_num,
                    categorical_data=all_cat,
                    feature_mask=all_mask,
                    apply_feature_dropout=False, # MFM is off, so don't drop features
                    dropout_rate=0.0
                )
                # --- END V5.1 FIX ---
                
                (anchor_emb, positive_emb, negative_emb) = torch.chunk(all_embeddings, 3, dim=0)

                (anchor_final, positive_final, negative_final) = online_semi_hard_triplet_mining(
                    anchor_emb, positive_emb, negative_emb, TRIPLET_MARGIN
                )
                
                loss_triplet = torch.tensor(0.0, device=device)
                if anchor_final is not None and anchor_final.shape[0] > 0:
                    loss_triplet = triplet_loss_fn(anchor_final, positive_final, negative_final)

                # --- V5.1 FIX: MFM Loss is 0 by definition ---
                loss_mfm = torch.tensor(0.0, device=device)
                
                # We can skip the 'calculate_mfm_loss' call entirely
                # if W_MFM_S2 > 0 and mfm_predictions is not None:
                #    ... (code removed) ...

                total_batch_loss = loss_triplet + (W_MFM_S2 * loss_mfm) # W_MFM_S2 is 0.0
                # --- END V5.1 FIX ---

            if torch.isnan(total_batch_loss):
                 logger.error(f"NaN loss detected at Stage 2, Epoch {epoch+1}. Stopping training.")
                 return
            
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer_stage2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer_stage2)
            scaler.update()

            total_loss_s2 += total_batch_loss.item()
            total_triplet_s2 += loss_triplet.item()
            total_mfm_s2 += loss_mfm.item() # This will just add 0.0
            num_batches_s2 += 1
            if num_batches_s2 > 0:
                pbar.set_postfix(
                    Loss=f"{(total_loss_s2/num_batches_s2):.4f}", 
                    Triplet=f"{(total_triplet_s2/num_batches_s2):.4f}",
                    MFM=f"{(total_mfm_s2/num_batches_s2):.4f}" # Will show 0.0
                )

    logger.info("--- Stage 2 Complete ---")

    # --- Save the Final Model ---
    abs_save_path = os.path.join(project_root, MODEL_SAVE_PATH)
    os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)
    logger.info(f"Saving final V4.2 trained model state_dict to {abs_save_path}...")
    
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(model_state, abs_save_path)
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    try:
        train_v4_model()
    except Exception as e:
        print("--- FATAL: RUNTIME ERROR ---")
        print(f"The script failed during execution. Error: {e}")
        traceback.print_exc()
        sys.exit(1)