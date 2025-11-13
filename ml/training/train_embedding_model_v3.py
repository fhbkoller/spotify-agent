# ml/training/train_embedding_model_v3.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logging import logger

# --- Configuration ---
FMA_DATASET_PATH = 'ml/data/triplets.csv'
SPECIFICITY_DATASET_PATH = 'ml/data/synthetic_specificity_v2.csv'
MODEL_SAVE_PATH = 'ml/data/models/v3.0_multitask_model_final.pth'

NUM_GENRES = 252 # From build_jamendo_lookups.py
NUM_KEYS = 12
EMBEDDING_DIM = 12

# --- Hyperparameters ---
# --- Reset to our proven "fast" batch size ---
BATCH_SIZE_STAGE1 = 2048 
BATCH_SIZE_STAGE2 = 1024 

EPOCHS_STAGE1 = 100
LEARNING_RATE_STAGE1 = 1e-4 
TRIPLET_MARGIN_STAGE1 = 0.2
W_GENRE_S1 = 0.5
W_KEY_S1 = 1.0
W_MODE_S1 = 0.5

EPOCHS_STAGE2 = 100
LEARNING_RATE_STAGE2 = 1e-5
TRIPLET_MARGIN_STAGE2 = 0.2
W_GENRE_S2 = 0.1
W_KEY_S2 = 1.5
W_MODE_S2 = 1.0

GRAD_CLIP_NORM = 1.0

BASE_FEATURE_COLUMNS = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'speechiness', 'valence', 'tempo', 'loudness', # 0-8
    'key', 'mode' # 9, 10
]
KEY_INDEX = 9
MODE_INDEX = 10
NUM_RAW_FEATURES = 11

# --- V3.0: "Pre-Baked CPU" Dataset Class ---
class SongDatasetV3(Dataset):
    """
    Pre-bakes all tensors and stores them in CPU RAM.
    This allows for multi-processing (num_workers > 0) in the DataLoader.
    """
    def __init__(self, csv_path):
        logger.info(f"Loading V3 dataset from {csv_path}...")
        data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(data)} triplets.")

        # --- Filter ---
        initial_len = len(data)
        data = data[data['anchor_genre_label'] != -1].reset_index(drop=True)
        filtered_len = len(data)
        if initial_len != filtered_len:
            logger.info(f"Filtered out {initial_len - filtered_len} tracks with no genre (label == -1).")
        
        self.dataset_len = len(data)

        # --- Convert to NumPy for efficient iteration ---
        logger.info("Converting to NumPy arrays...")
        anchor_cols = [f'anchor_feat_{i}' for i in range(NUM_RAW_FEATURES)]
        positive_cols = [f'positive_feat_{i}' for i in range(NUM_RAW_FEATURES)]
        negative_cols = [f'negative_feat_{i}' for i in range(NUM_RAW_FEATURES)]
        
        anchor_features_np = data[anchor_cols].to_numpy(dtype=np.float32)
        positive_features_np = data[positive_cols].to_numpy(dtype=np.float32)
        negative_features_np = data[negative_cols].to_numpy(dtype=np.float32)
        genre_labels_np = data['anchor_genre_label'].to_numpy(dtype=np.int64)
        
        del data # Release DataFrame memory
        
        # --- Pre-bake all tensors (in CPU RAM) ---
        logger.info(f"Pre-baking all {self.dataset_len} items into CPU tensors...")
        self.anchors = []
        self.positives = []
        self.negatives = []
        self.genre_labels = []
        self.key_labels = []
        self.mode_labels = []

        for i in tqdm(range(self.dataset_len), desc="Pre-processing dataset"):
            anchor_tensor, key_label, mode_label = self._process_features(anchor_features_np[i])
            self.anchors.append(anchor_tensor)
            self.key_labels.append(torch.tensor(key_label, dtype=torch.long))
            self.mode_labels.append(torch.tensor([mode_label], dtype=torch.float32))
            
            positive_tensor, _, _ = self._process_features(positive_features_np[i])
            self.positives.append(positive_tensor)
            
            negative_tensor, _, _ = self._process_features(negative_features_np[i])
            self.negatives.append(negative_tensor)

            self.genre_labels.append(torch.tensor(genre_labels_np[i], dtype=torch.long))
            
        logger.info("Pre-baking complete. NumPy arrays released.")
        del anchor_features_np, positive_features_np, negative_features_np, genre_labels_np

        # --- DO NOT MOVE TO GPU HERE ---
        logger.info("Dataset successfully loaded into CPU RAM.")

    def __len__(self):
        return self.dataset_len

    def _process_features(self, raw_features: np.ndarray):
        key_label = int(raw_features[KEY_INDEX])
        mode_label = raw_features[MODE_INDEX] # float32
        key_val = raw_features[KEY_INDEX]
        key_sin = np.sin(2 * np.pi * key_val / 12.0)
        key_cos = np.cos(2 * np.pi * key_val / 12.0)
        
        final_features = np.concatenate(
            (raw_features[:KEY_INDEX], [mode_label, key_sin, key_cos])
        ).astype(np.float32)
        
        feature_tensor = torch.from_numpy(final_features)
        return feature_tensor, key_label, mode_label

    def __getitem__(self, idx):
        # --- Blazing-fast indexing from CPU RAM ---
        return (
            self.anchors[idx],
            self.positives[idx],
            self.negatives[idx],
            self.genre_labels[idx],
            self.key_labels[idx],
            self.mode_labels[idx]
        )


# --- V3.0: Model Architecture (Unchanged, refactored for single pass) ---
class EmbeddingTrunk(nn.Module):
    def __init__(self, input_dim=12, embedding_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, embedding_dim)
        self.gate_fc1 = nn.Linear(input_dim, 32)
        self.gate_fc2 = nn.Linear(32, embedding_dim)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        m = self.leaky_relu(self.batch_norm1(self.fc1(x)))
        m = self.dropout(m)
        m = self.leaky_relu(self.batch_norm2(self.fc2(m)))
        m = self.dropout(m)
        main_output = self.leaky_relu(self.batch_norm3(self.fc3(m)))
        g = self.leaky_relu(self.gate_fc1(x))
        gate = self.sigmoid(self.gate_fc2(g))
        gated_trunk_output = main_output * gate
        return gated_trunk_output

class MultiTaskEmbeddingModel_V3(nn.Module):
    def __init__(self, input_dim=12, embedding_dim=12, num_genres=252, num_keys=12, trunk_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_genres = num_genres
        self.num_keys = num_keys
        self.trunk = EmbeddingTrunk(input_dim=input_dim, embedding_dim=trunk_dim)
        self.embedding_head = nn.Linear(trunk_dim, embedding_dim)
        self.genre_head = nn.Sequential(
            nn.Linear(trunk_dim, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(32, num_genres)
        )
        self.key_head = nn.Sequential(
            nn.Linear(trunk_dim, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(32, num_keys)
        )
        self.mode_head = nn.Sequential(
            nn.Linear(trunk_dim, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        trunk_output = self.trunk(x)
        emb_raw = self.embedding_head(trunk_output)
        embedding = F.normalize(emb_raw, p=2, dim=1)
        genre_logits = self.genre_head(trunk_output)
        key_logits = self.key_head(trunk_output)
        mode_logits = self.mode_head(trunk_output)
        return embedding, genre_logits, key_logits, mode_logits

# --- Training Loop ---
def train_v3_model():
    logger.info("--- Starting Training: V3.0 Multi-Task (4-Head) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == 'cpu':
        logger.warning("Running on CPU. Optimizations (AMP, Compile) will be disabled.")
    
    # --- JIT Compilation ---
    model = MultiTaskEmbeddingModel_V3(
        input_dim=12,
        embedding_dim=EMBEDDING_DIM,
        num_genres=NUM_GENRES,
        num_keys=NUM_KEYS,
        trunk_dim=64
    ).to(device)
    
    if device.type == 'cuda':
        logger.info("Applying torch.compile() for JIT optimization...")
        model = torch.compile(model) # This will work in WSL
    
    logger.info("Initializing V3.0 Model...")
    logger.info(f"Model configured with NUM_GENRES = {NUM_GENRES}")
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN_STAGE1, p=2)
    genre_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    key_loss_fn = nn.CrossEntropyLoss()
    mode_loss_fn = nn.BCEWithLogitsLoss()
    
    # --- Automatic Mixed Precision (AMP) Scaler ---
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    logger.info(f"Initializing GradScaler (Enabled: {device.type == 'cuda'})...")

    # --- USE CPU WORKERS ---
    num_workers = min(8, os.cpu_count()) if os.cpu_count() else 0 # Use 8 workers
    pin_memory = True # Must be True for async transfer
    logger.info(f"Setting num_workers = {num_workers} and pin_memory = True.")
    
    # --- Stage 1: Pre-training on Jamendo V2 ---
    logger.info("\n--- Stage 1: Pre-training on Jamendo (V2 Dataset) ---")
    logger.info(f"Loading Jamendo dataset from {FMA_DATASET_PATH}...")
    try:
        fma_dataset = SongDatasetV3(FMA_DATASET_PATH) # No device passed
        fma_dataloader = DataLoader(
            fma_dataset, 
            batch_size=BATCH_SIZE_STAGE1, # 2048
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        logger.info(f"Loaded {len(fma_dataset)} Jamendo triplets (after filtering).")
    except Exception as e:
        logger.error(f"Error loading Jamendo dataset: {e}")
        return

    optimizer_stage1 = optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE1)
    logger.info(f"  Epochs: {EPOCHS_STAGE1} | LR: {LEARNING_RATE_STAGE1} | Batch: {BATCH_SIZE_STAGE1}")
    logger.info(f"  Loss Weights: Triplet(1.0), Genre({W_GENRE_S1}), Key({W_KEY_S1}), Mode({W_MODE_S1})")
    
    model.train()
    for epoch in range(EPOCHS_STAGE1):
        total_loss_s1, total_triplet_s1, total_genre_s1, total_key_s1, total_mode_s1 = 0.0, 0.0, 0.0, 0.0, 0.0
        num_batches_s1 = 0
        
        for anchor, positive, negative, genre_labels, key_labels, mode_labels in tqdm(fma_dataloader, desc=f"Stage 1 - Epoch {epoch+1}/{EPOCHS_STAGE1}"):
            
            # --- ASYNC TRANSFER TO GPU ---
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            genre_labels = genre_labels.to(device, non_blocking=True)
            key_labels = key_labels.to(device, non_blocking=True)
            mode_labels = mode_labels.to(device, non_blocking=True)

            combined_input = torch.cat((anchor, positive, negative), dim=0)
            
            optimizer_stage1.zero_grad()
            
            # --- 1. AMP: autocast context ---
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                all_embs, all_genres, all_keys, all_modes = model(combined_input)

                anchor_emb, positive_emb, negative_emb = torch.chunk(all_embs, 3, dim=0)
                anchor_genre_logits, _, _ = torch.chunk(all_genres, 3, dim=0)
                anchor_key_logits, _, _ = torch.chunk(all_keys, 3, dim=0)
                anchor_mode_logits, _, _ = torch.chunk(all_modes, 3, dim=0)

                loss_triplet = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                loss_genre = genre_loss_fn(anchor_genre_logits, genre_labels)
                loss_key = key_loss_fn(anchor_key_logits, key_labels)
                loss_mode = mode_loss_fn(anchor_mode_logits, mode_labels)
                
                total_batch_loss = (loss_triplet +
                                    (W_GENRE_S1 * loss_genre) +
                                    (W_KEY_S1 * loss_key) +
                                    (W_MODE_S1 * loss_mode))

            if torch.isnan(total_batch_loss):
                logger.error(f"NaN loss detected at Stage 1, Epoch {epoch+1}. Stopping training.")
                return

            # --- 1. AMP: Use scaler to backward pass ---
            scaler.scale(total_batch_loss).backward()
            
            # Clip gradients
            scaler.unscale_(optimizer_stage1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            # --- 1. AMP: Use scaler to step ---
            scaler.step(optimizer_stage1)
            scaler.update()

            total_loss_s1 += total_batch_loss.item()
            total_triplet_s1 += loss_triplet.item()
            total_genre_s1 += loss_genre.item()
            total_key_s1 += loss_key.item()
            total_mode_s1 += loss_mode.item()
            num_batches_s1 += 1
            
        logger.info(f"Stage 1 - Epoch {epoch+1}: Total Loss={(total_loss_s1/num_batches_s1):.4f} "
                    f"[Triplet={(total_triplet_s1/num_batches_s1):.4f}, Genre={(total_genre_s1/num_batches_s1):.4f}, "
                    f"Key={(total_key_s1/num_batches_s1):.4f}, Mode={(total_mode_s1/num_batches_s1):.4f}]")
    logger.info("--- Stage 1 Complete ---")

    # --- Stage 2: Fine-tuning on Specificity V2 ---
    logger.info("\n--- Stage 2: Fine-tuning on Specificity (V2 Dataset) ---")
    logger.info(f"Loading Specificity dataset from {SPECIFICITY_DATASET_PATH}...")
    try:
        spec_dataset = SongDatasetV3(SPECIFICITY_DATASET_PATH) # No device passed
        spec_dataloader = DataLoader(
            spec_dataset, 
            batch_size=BATCH_SIZE_STAGE2, # 1024
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        logger.info(f"Loaded {len(spec_dataset)} Specificity triplets (after filtering).")
    except Exception as e:
        logger.error(f"Error loading Specificity dataset: {e}")
        return

    optimizer_stage2 = optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE2)
    triplet_loss_fn.margin = TRIPLET_MARGIN_STAGE2
    
    logger.info(f"  Epochs: {EPOCHS_STAGE2} | LR: {LEARNING_RATE_STAGE2} | Batch: {BATCH_SIZE_STAGE2}")
    logger.info(f"  Loss Weights: Triplet(1.0), Genre({W_GENRE_S2}), Key({W_KEY_S2}), Mode({W_MODE_S2})")

    model.train()
    for epoch in range(EPOCHS_STAGE2):
        total_loss_s2, total_triplet_s2, total_genre_s2, total_key_s2, total_mode_s2 = 0.0, 0.0, 0.0, 0.0, 0.0
        num_batches_s2 = 0
        
        for anchor, positive, negative, genre_labels, key_labels, mode_labels in tqdm(spec_dataloader, desc=f"Stage 2 - Epoch {epoch+1}/{EPOCHS_STAGE2}"):
            
            # --- ASYNC TRANSFER TO GPU ---
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            genre_labels = genre_labels.to(device, non_blocking=True)
            key_labels = key_labels.to(device, non_blocking=True)
            mode_labels = mode_labels.to(device, non_blocking=True)

            combined_input = torch.cat((anchor, positive, negative), dim=0)
            
            optimizer_stage2.zero_grad()
            
            # --- 1. AMP: autocast context ---
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                all_embs, all_genres, all_keys, all_modes = model(combined_input)

                anchor_emb, positive_emb, negative_emb = torch.chunk(all_embs, 3, dim=0)
                anchor_genre_logits, _, _ = torch.chunk(all_genres, 3, dim=0)
                anchor_key_logits, _, _ = torch.chunk(all_keys, 3, dim=0)
                anchor_mode_logits, _, _ = torch.chunk(all_modes, 3, dim=0)

                
                loss_triplet = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                loss_genre = genre_loss_fn(anchor_genre_logits, genre_labels)
                loss_key = key_loss_fn(anchor_key_logits, key_labels)
                loss_mode = mode_loss_fn(anchor_mode_logits, mode_labels)
                
                total_batch_loss = (loss_triplet +
                                    (W_GENRE_S2 * loss_genre) +
                                    (W_KEY_S2 * loss_key) +
                                    (W_MODE_S2 * loss_mode))

            if torch.isnan(total_batch_loss):
                 logger.error(f"NaN loss detected at Stage 2, Epoch {epoch+1}. Stopping training.")
                 return
            
            # --- 1. AMP: Use scaler ---
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer_stage2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer_stage2)
            scaler.update()

            total_loss_s2 += total_batch_loss.item()
            total_triplet_s2 += loss_triplet.item()
            total_genre_s2 += loss_genre.item()
            total_key_s2 += loss_key.item()
            total_mode_s2 += loss_mode.item()
            num_batches_s2 += 1
            
        logger.info(f"Stage 2 - Epoch {epoch+1}: Total Loss={(total_loss_s2/num_batches_s2):.4f} "
                    f"[Triplet={(total_triplet_s2/num_batches_s2):.4f}, Genre={(total_genre_s2/num_batches_s2):.4f}, "
                    f"Key={(total_key_s2/num_batches_s2):.4f}, Mode={(total_mode_s2/num_batches_s2):.4f}]")

    logger.info("--- Stage 2 Complete ---")

    # --- Save the Final Model ---
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    logger.info(f"Saving final V3.0 trained model state_dict to {MODEL_SAVE_PATH}...")
    torch.save(model.to('cpu').state_dict(), MODEL_SAVE_PATH)
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    train_v3_model()