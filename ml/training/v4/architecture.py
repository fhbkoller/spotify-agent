# ml/training/v4/architecture.py
# (Refactored V5.0: THE DEFINITIVE PATHING AND CONSTANT FIX)

import os
import sys

# --- V5.0 ROBUST PATHING FIX ---
try:
    # Get the directory of *this* script (ml/training/v4)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels to the project root
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    # Explicitly add the 'src' directory to the path
    src_path = os.path.join(project_root, 'src')

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(1, src_path) # Add src dir
except NameError:
    # Fallback if __file__ is not defined
    project_root = os.path.abspath('.')
    src_path = os.path.join(project_root, 'src')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(1, src_path)
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
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Optional, Tuple

from src.utils.logging import logger

# --- V4.0 Model Hyperparameters ---
NUM_KEYS = 12
NUM_MODES = 2

D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
OUTPUT_EMBEDDING_DIM = 128 # 128-dim output

# --- V5.0 FIX: DEFINE CONSTANTS HERE ---
# This is the single source of truth for the model's input structure.
CATEGORICAL_FEATURE_KEYS_ORDERED = ['genre', 'key', 'mode']
NUMERICAL_FEATURE_KEYS_ORDERED = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 
    'speechiness', 'valence', 'tempo', 'loudness'
]
ALL_FEATURE_KEYS_ORDERED = CATEGORICAL_FEATURE_KEYS_ORDERED + NUMERICAL_FEATURE_KEYS_ORDERED
# --- END V5.0 FIX ---


class FeatureTokenizer(nn.Module):
    """
    (Refactored V4.2)
    Implements correct FT-Transformer numerical tokenization using
    a single embedding layer for identities and a shared projection for values.
    """
    def __init__(self, 
                 genre_to_index_map: Dict[str, int],
                 d_model: int
                ):
        super().__init__()
        self.d_model = d_model

        # --- Tokenizers for Categorical Features ---
        self.num_genres = len(genre_to_index_map)
        self.genre_embed = nn.Embedding(self.num_genres, d_model, padding_idx=0)
        self.key_tokenizer = nn.Linear(2, d_model) # 2D sin/cos -> D_MODEL
        self.mode_embed = nn.Embedding(NUM_MODES, d_model)

        # --- Tokenizers for Numerical Features ---
        self.numerical_projection = nn.Linear(1, d_model)
        
        # V5.0 FIX: Use the locally defined constant
        self.num_numerical_features = len(NUMERICAL_FEATURE_KEYS_ORDERED) 
        self.numerical_identity_embeddings = nn.Embedding(self.num_numerical_features, d_model)
        
        self.numerical_layer_norm = nn.LayerNorm(d_model)
        
    def _encode_key_batch(self, key_val_batch: torch.Tensor) -> torch.Tensor:
        """Encodes a BATCH of key values (0-11) into a (B, 2) sin/cos tensor."""
        key_sin = torch.sin(2 * math.pi * key_val_batch / 12.0)
        key_cos = torch.cos(2 * math.pi * key_val_batch / 12.0)
        return torch.stack([key_sin, key_cos], dim=1)

    def forward(self, 
              numerical_data: torch.Tensor,    # (B, 9)
              categorical_data: torch.Tensor   # (B, 3) -> [genre_idx, key_val, mode_val]
             ) -> torch.Tensor:
        
        batch_size = numerical_data.shape[0]
        
        # --- 1. Process Categorical Tokens ---
        genre_tokens = self.genre_embed(categorical_data[:, 0]).unsqueeze(1)
        key_sincos = self._encode_key_batch(categorical_data[:, 1].float())
        key_tokens = self.key_tokenizer(key_sincos).unsqueeze(1)
        mode_tokens = self.mode_embed(categorical_data[:, 2]).unsqueeze(1)

        # --- 2. Process Numerical Tokens ---
        numerical_values = numerical_data.unsqueeze(-1)
        value_projections = self.numerical_projection(numerical_values)
        numerical_indices = torch.arange(self.num_numerical_features, device=numerical_data.device)
        identity_embeddings = self.numerical_identity_embeddings(numerical_indices)
        identity_embeddings = identity_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        numerical_tokens = value_projections + identity_embeddings
        numerical_tokens = self.numerical_layer_norm(numerical_tokens)

        # --- 3. Stack All Tokens ---
        all_tokens = torch.cat(
            [genre_tokens, key_tokens, mode_tokens, numerical_tokens],
            dim=1
        )
        return all_tokens


# --- MFM Head Module ---
class MFMHead(nn.Module):
    """
    Decodes the 12 feature tokens back into their original feature values.
    """
    def __init__(self, d_model: int, num_genres: int):
        super().__init__()
        
        self.genre_head = nn.Linear(d_model, num_genres)
        self.key_head = nn.Linear(d_model, NUM_KEYS)
        self.mode_head = nn.Linear(d_model, NUM_MODES)
        
        self.numerical_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, feature_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        genre_logits = self.genre_head(feature_tokens[:, 0])
        key_logits = self.key_head(feature_tokens[:, 1])
        mode_logits = self.mode_head(feature_tokens[:, 2])
        numerical_preds = self.numerical_head(feature_tokens[:, 3:]).squeeze(-1)
        
        return genre_logits, key_logits, mode_logits, numerical_preds

class VibeFTTransformer(nn.Module):
    """
    V4.2 FT-Transformer for Vibe Embedding.
    """
    def __init__(self, 
                 genre_to_index_map: Dict[str, int],
                 d_model: int, 
                 n_head: int, 
                 num_layers: int, 
                 dim_feedforward: int, 
                 output_dim: int, 
                 dropout: float = 0.1
                ):
        super().__init__()
        
        self.num_genres = len(genre_to_index_map)
        
        self.tokenizer = FeatureTokenizer(
            genre_to_index_map=genre_to_index_map,
            d_model=d_model
        )
        
        self.vibe_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, output_dim)
        )
        
        self.mfm_head = MFMHead(d_model, self.num_genres)

    def forward(self, 
              numerical_data: torch.Tensor,
              categorical_data: torch.Tensor,
              feature_mask: torch.Tensor,
              apply_feature_dropout: bool = False,
              dropout_rate: float = 0.25
             ) -> Tuple[torch.Tensor, Tuple | None, torch.Tensor | None]:
        
        batch_size = numerical_data.shape[0]
        
        feature_tokens = self.tokenizer(numerical_data, categorical_data)
        
        src_key_padding_mask = ~feature_mask
        
        dropout_and_padding_mask = src_key_padding_mask.clone()
        mfm_dropout_mask = None

        if apply_feature_dropout and self.training:
            dropout_mask = (torch.rand(batch_size, 12, device=feature_tokens.device) > dropout_rate)
            # dropout_mask is True for *kept* tokens
            # We want to mask out (True) anything that was *either*
            # originally padding (src_key_padding_mask) OR is now dropped (~dropout_mask)
            dropout_and_padding_mask = src_key_padding_mask | (~dropout_mask)
            
            # The MFM mask: True for tokens that were *dropped* AND *originally present*
            mfm_dropout_mask = feature_mask & (~dropout_mask)

        
        vibe_tokens = self.vibe_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([vibe_tokens, feature_tokens], dim=1)
        
        vibe_token_pad = torch.full((batch_size, 1), False, device=dropout_and_padding_mask.device)
        full_padding_mask = torch.cat([vibe_token_pad, dropout_and_padding_mask], dim=1)
        
        transformer_output = self.transformer_encoder(
            src=all_tokens,
            src_key_padding_mask=full_padding_mask
        )
        
        vibe_token_output = transformer_output[:, 0]
        feature_outputs = transformer_output[:, 1:]
        
        final_embedding_raw = self.embedding_head(vibe_token_output)
        final_embedding = F.normalize(final_embedding_raw, p=2, dim=1)
        
        mfm_predictions = None
        if (apply_feature_dropout and self.training) or (not self.training and feature_outputs.numel() > 0): # Also get preds for eval
             mfm_predictions = self.mfm_head(feature_outputs)

        return final_embedding, mfm_predictions, mfm_dropout_mask