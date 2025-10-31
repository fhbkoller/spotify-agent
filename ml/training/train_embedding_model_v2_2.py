# ml/training/train_embedding_model_v2_2.py

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
import re # Added for parsing in SongDataset if needed later

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logging import logger

# --- Configuration ---
FMA_DATASET_PATH = 'ml/data/fma_triplets.csv'
SPECIFICITY_DATASET_PATH = 'ml/data/synthetic_specificity_v1.csv'
MODEL_SAVE_PATH = 'ml/data/models/v2_2_multitask_model_final.pth' # Final model path
NUM_GENRES = 15 # Number of top-level genres in FMA dataset

# --- Hyperparameters ---
# Stage 1
EPOCHS_STAGE1 = 100
LEARNING_RATE_STAGE1 = 1e-4
BATCH_SIZE_STAGE1 = 128
TRIPLET_MARGIN_STAGE1 = 0.2
ALPHA = 0.5             # Weight for genre loss in Stage 1
GRAD_CLIP_NORM_STAGE1 = 1.0

# Stage 2
EPOCHS_STAGE2 = 100
LEARNING_RATE_STAGE2 = 1e-5 # Keep reduced LR
BATCH_SIZE_STAGE2 = 64
TRIPLET_MARGIN_STAGE2 = 0.2
# --- FIX: Re-enable genre loss in Stage 2 with small weight ---
BETA = 0.01
GRAD_CLIP_NORM_STAGE2 = 1.0

EMBEDDING_DIM = 12      # Final output dimension (8 num + mode + key_sin + key_cos)

# Define the 11 base feature names in the order they should appear before encoding
BASE_FEATURE_COLUMNS = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'speechiness', 'valence', 'tempo', 'loudness', # Numerical 0-8
    'key', 'mode' # Categorical 9, 10
]
KEY_INDEX = BASE_FEATURE_COLUMNS.index('key') # Should be 9
MODE_INDEX = BASE_FEATURE_COLUMNS.index('mode') # Should be 10

# --- Dataset Class (Unchanged) ---
class SongDataset(Dataset):
    """
    Loads song triplets from CSV. Detects format (FMA vs Specificity)
    and applies Sin/Cos encoding for the key.
    """
    def __init__(self, csv_path):
        logger.info(f"Loading dataset from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.data)} triplets.")

        self.is_fma_data = 'anchor_genre_label' in self.data.columns
        if f'anchor_{BASE_FEATURE_COLUMNS[0]}' in self.data.columns:
            self.format_type = 'fma_descriptive'
            logger.info("Detected FMA dataset format (descriptive names, with genre labels).")
            if not self.is_fma_data:
                logger.warning("CSV has descriptive names but is missing 'anchor_genre_label'. Treating as Specificity data.")
                self.is_fma_data = False
        elif 'anchor_feat_0' in self.data.columns:
            self.format_type = 'specificity_indexed'
            logger.info("Detected Specificity dataset format (indexed names, no genre labels).")
            self.is_fma_data = False
        else:
            raise ValueError(f"Could not determine dataset format from columns in {csv_path}")

    def __len__(self):
        return len(self.data)

    def _process_features(self, row, prefix):
        """Extracts features based on detected format, applies sin/cos, returns tensor."""
        raw_features = []
        if self.format_type == 'fma_descriptive':
            raw_features = [row[f'{prefix}_{name}'] for name in BASE_FEATURE_COLUMNS]
        elif self.format_type == 'specificity_indexed':
            raw_features = [row[f'{prefix}_feat_{i}'] for i in range(len(BASE_FEATURE_COLUMNS))]
        else:
             raise RuntimeError("Unknown format_type in _process_features")

        key = raw_features[KEY_INDEX]
        mode = raw_features[MODE_INDEX]

        key_sin = np.sin(2 * np.pi * key / 12.0)
        key_cos = np.cos(2 * np.pi * key / 12.0)

        # Final 12-feature vector
        final_features = raw_features[:KEY_INDEX] + [mode, key_sin, key_cos]

        return torch.tensor(final_features, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor = self._process_features(row, 'anchor')
        positive = self._process_features(row, 'positive')
        negative = self._process_features(row, 'negative')
        label = torch.tensor(row['anchor_genre_label'], dtype=torch.long) if self.is_fma_data else torch.tensor(-1, dtype=torch.long)
        return anchor, positive, negative, label

# --- Model Architecture (Unchanged) ---
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

class MultiTaskEmbeddingModel(nn.Module):
    def __init__(self, input_dim=12, embedding_dim=12, num_genres=15, trunk_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_genres = num_genres
        self.trunk = EmbeddingTrunk(input_dim=input_dim, embedding_dim=trunk_dim)
        self.embedding_head = nn.Linear(trunk_dim, embedding_dim)
        self.genre_head = nn.Sequential(
            nn.Linear(trunk_dim, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(32, num_genres)
        )

    def forward(self, anchor, positive=None, negative=None):
        anchor_trunk = self.trunk(anchor)
        anchor_emb_raw = self.embedding_head(anchor_trunk)
        anchor_emb = F.normalize(anchor_emb_raw, p=2, dim=1)
        anchor_genre_logits = self.genre_head(anchor_trunk)
        if positive is None or negative is None:
            return anchor_emb, anchor_genre_logits
        positive_trunk = self.trunk(positive)
        positive_emb_raw = self.embedding_head(positive_trunk)
        positive_emb = F.normalize(positive_emb_raw, p=2, dim=1)
        negative_trunk = self.trunk(negative)
        negative_emb_raw = self.embedding_head(negative_trunk)
        negative_emb = F.normalize(negative_emb_raw, p=2, dim=1)
        return anchor_emb, positive_emb, negative_emb, anchor_genre_logits

# --- Training Loop ---
def train_multitask_model():
    logger.info("--- Starting Training: V2.2 Multi-Task (Sin/Cos) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = MultiTaskEmbeddingModel(input_dim=12, embedding_dim=EMBEDDING_DIM, num_genres=NUM_GENRES, trunk_dim=64).to(device)
    logger.info("Initializing V2.2 Multi-Task Model...")
    triplet_loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN_STAGE1, p=2)
    genre_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # ignore_index handles dummy labels

    # --- Stage 1: Pre-training on FMA ---
    logger.info("\n--- Stage 1: Pre-training on FMA ---")
    logger.info(f"Loading FMA dataset from {FMA_DATASET_PATH}...")
    try:
        fma_dataset = SongDataset(FMA_DATASET_PATH)
        num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
        fma_dataloader = DataLoader(fma_dataset, batch_size=BATCH_SIZE_STAGE1, shuffle=True, num_workers=num_workers, pin_memory=True if device=='cuda' else False)
        logger.info(f"Loaded {len(fma_dataset)} FMA triplets.")
    except Exception as e:
        logger.error(f"Error loading FMA dataset: {e}")
        return

    optimizer_stage1 = optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE1)
    logger.info(f"  Epochs: {EPOCHS_STAGE1} | LR: {LEARNING_RATE_STAGE1} | Batch: {BATCH_SIZE_STAGE1} | Margin: {TRIPLET_MARGIN_STAGE1} | Alpha: {ALPHA} | Grad Clip: {GRAD_CLIP_NORM_STAGE1}")
    model.train()
    for epoch in range(EPOCHS_STAGE1):
        total_loss_s1, total_triplet_loss_s1, total_genre_loss_s1, num_batches_s1 = 0.0, 0.0, 0.0, 0
        for anchor, positive, negative, genre_labels in tqdm(fma_dataloader, desc=f"Stage 1 - Epoch {epoch+1}/{EPOCHS_STAGE1}"):
            anchor, positive, negative, genre_labels = anchor.to(device), positive.to(device), negative.to(device), genre_labels.to(device)
            optimizer_stage1.zero_grad()
            anchor_emb, positive_emb, negative_emb, anchor_genre_logits = model(anchor, positive, negative)
            loss_triplet = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            loss_genre = genre_loss_fn(anchor_genre_logits, genre_labels)
            total_batch_loss = loss_triplet + (ALPHA * loss_genre)

            if torch.isnan(total_batch_loss):
                logger.error(f"NaN loss detected at Stage 1, Epoch {epoch+1}. Stopping training.")
                return

            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM_STAGE1)
            optimizer_stage1.step()

            total_loss_s1 += total_batch_loss.item()
            total_triplet_loss_s1 += loss_triplet.item()
            total_genre_loss_s1 += loss_genre.item()
            num_batches_s1 += 1
        avg_loss = total_loss_s1 / num_batches_s1 if num_batches_s1 > 0 else 0
        avg_triplet_loss = total_triplet_loss_s1 / num_batches_s1 if num_batches_s1 > 0 else 0
        avg_genre_loss = total_genre_loss_s1 / num_batches_s1 if num_batches_s1 > 0 else 0
        logger.info(f"Stage 1 - Epoch {epoch+1}/{EPOCHS_STAGE1}: Avg Total Loss={avg_loss:.6f}, Avg Triplet Loss={avg_triplet_loss:.6f}, Avg Genre Loss={avg_genre_loss:.6f}")
    logger.info("--- Stage 1 Complete ---")

    # --- Stage 2: Fine-tuning on Specificity ---
    logger.info("\n--- Stage 2: Fine-tuning on Specificity ---")
    logger.info(f"Loading Specificity dataset from {SPECIFICITY_DATASET_PATH}...")
    try:
        spec_dataset = SongDataset(SPECIFICITY_DATASET_PATH)
        num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
        spec_dataloader = DataLoader(spec_dataset, batch_size=BATCH_SIZE_STAGE2, shuffle=True, num_workers=num_workers, pin_memory=True if device=='cuda' else False)
        logger.info(f"Loaded {len(spec_dataset)} Specificity triplets.")
    except Exception as e:
        logger.error(f"Error loading Specificity dataset: {e}")
        return

    # Create a NEW optimizer instance for Stage 2
    optimizer_stage2 = optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE2)
    triplet_loss_fn.margin = TRIPLET_MARGIN_STAGE2
    # --- FIX: Update log message to show BETA ---
    logger.info(f"  Epochs: {EPOCHS_STAGE2} | LR: {LEARNING_RATE_STAGE2} | Batch: {BATCH_SIZE_STAGE2} | Margin: {TRIPLET_MARGIN_STAGE2} | Grad Clip: {GRAD_CLIP_NORM_STAGE2} | Beta: {BETA}")

    model.train()
    for epoch in range(EPOCHS_STAGE2):
        total_loss_s2, total_triplet_loss_s2, total_genre_loss_s2, num_batches_s2 = 0.0, 0.0, 0.0, 0
        for anchor, positive, negative, dummy_labels in tqdm(spec_dataloader, desc=f"Stage 2 - Epoch {epoch+1}/{EPOCHS_STAGE2}"):
            anchor, positive, negative, dummy_labels = anchor.to(device), positive.to(device), negative.to(device), dummy_labels.to(device)
            optimizer_stage2.zero_grad()
            anchor_emb, positive_emb, negative_emb, anchor_genre_logits = model(anchor, positive, negative)

            loss_triplet = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            # --- FIX: Re-enable genre loss calculation ---
            loss_genre = genre_loss_fn(anchor_genre_logits, dummy_labels)
            # --- FIX: Use BETA to combine losses ---
            total_batch_loss = loss_triplet + (BETA * loss_genre)

            if torch.isnan(total_batch_loss):
                 logger.error(f"NaN loss detected at Stage 2, Epoch {epoch+1}. Stopping training.")
                 logger.error(f"Triplet Loss: {loss_triplet.item()}, Genre Loss: {loss_genre.item()}")
                 return

            total_batch_loss.backward()
            # --- FIX: Add gradient clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM_STAGE2)
            optimizer_stage2.step()

            total_loss_s2 += total_batch_loss.item()
            total_triplet_loss_s2 += loss_triplet.item()
            total_genre_loss_s2 += loss_genre.item() # Log the calculated genre loss
            num_batches_s2 += 1

        avg_loss = total_loss_s2 / num_batches_s2 if num_batches_s2 > 0 else 0
        avg_triplet_loss = total_triplet_loss_s2 / num_batches_s2 if num_batches_s2 > 0 else 0
        avg_genre_loss = total_genre_loss_s2 / num_batches_s2 if num_batches_s2 > 0 else 0
        logger.info(f"Stage 2 - Epoch {epoch+1}/{EPOCHS_STAGE2}: Avg Total Loss={avg_loss:.6f}, Avg Triplet Loss={avg_triplet_loss:.6f}, Avg Genre Loss={avg_genre_loss:.6f}") # Removed note about disabling

    logger.info("--- Stage 2 Complete ---")

    # --- Save the Final Model ---
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    logger.info(f"Saving final trained model state_dict to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info("Model saved successfully.")


if __name__ == "__main__":
    if sys.platform == 'win32':
        pass
    train_multitask_model()