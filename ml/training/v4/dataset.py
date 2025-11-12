# ml/training/v4/dataset.py
# (Refactored V5.0: Imports constants from architecture.py and fixes pathing)

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
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple

from src.utils.logging import logger

# --- V5.0 FIX: Import constants from architecture as the single source of truth ---
from ml.training.v4.architecture import (
    CATEGORICAL_FEATURE_KEYS_ORDERED,
    NUMERICAL_FEATURE_KEYS_ORDERED,
    ALL_FEATURE_KEYS_ORDERED
)
# --- END V5.0 FIX ---


class SongDatasetV4(Dataset):
    """
    Loads V4.0 triplet data and pre-processes it into tensors
    within the DataLoader workers (this is the fast way).
    """
    def __init__(self, parquet_path: str, genre_to_index_map: Dict[str, int]):
        self.parquet_path = parquet_path
        
        # V5.0 FIX: Use absolute path
        if not os.path.isabs(self.parquet_path):
            self.parquet_path = os.path.join(project_root, self.parquet_path)
            
        logger.info(f"Loading V4 dataset from {self.parquet_path}...")
        
        if not os.path.exists(self.parquet_path):
            logger.error(f"FATAL ERROR: Dataset file not found at {self.parquet_path}")
            raise FileNotFoundError(f"Dataset file not found at {self.parquet_path}")
            
        try:
            self.data = pd.read_parquet(self.parquet_path)
            logger.info(f"Successfully loaded {len(self.data)} triplets.")
        except Exception as e:
            logger.error(f"FATAL ERROR: Could not load parquet file {self.parquet_path}: {e}")
            raise e
        
        self.genre_to_index_map = genre_to_index_map
        self.unknown_genre_index = self.genre_to_index_map.get('genre::unknown', len(self.genre_to_index_map))

    def __len__(self):
        """Returns the total number of triplets in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Fetches a single triplet (anchor, positive, negative) at the given index.
        """
        try:
            row = self.data.iloc[idx]
            
            anchor_tensors = self._process_dict(row['anchor'])
            positive_tensors = self._process_dict(row['positive'])
            negative_tensors = self._process_dict(row['negative'])
            
            return (anchor_tensors, positive_tensors, negative_tensors)
            
        except IndexError:
            logger.error(f"IndexError: Index {idx} out of bounds for dataset with length {len(self.data)}")
            dummy = (torch.zeros(9), torch.zeros(3, dtype=torch.long), torch.zeros(12, dtype=torch.bool))
            return (dummy, dummy, dummy)
        except Exception as e:
            logger.error(f"Error getting item at index {idx}: {e}")
            dummy = (torch.zeros(9), torch.zeros(3, dtype=torch.long), torch.zeros(12, dtype=torch.bool))
            return (dummy, dummy, dummy)

    def _process_dict(self, feat_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a single feature dictionary into pre-processed tensors.
        This runs inside the DataLoader worker process.
        """
        
        # Use the imported constants
        numerical_data = torch.zeros(len(NUMERICAL_FEATURE_KEYS_ORDERED), dtype=torch.float32)
        categorical_data = torch.zeros(len(CATEGORICAL_FEATURE_KEYS_ORDERED), dtype=torch.long)
        feature_mask = torch.zeros(len(ALL_FEATURE_KEYS_ORDERED), dtype=torch.bool)
        
        # --- Process Categorical ---
        if 'genre' in feat_dict:
            genre_str = feat_dict['genre']
            genre_idx = self.genre_to_index_map.get(genre_str, self.unknown_genre_index)
            categorical_data[0] = genre_idx
            feature_mask[0] = True
            
        if 'key' in feat_dict:
            categorical_data[1] = int(feat_dict['key'])
            feature_mask[1] = True
            
        if 'mode' in feat_dict:
            categorical_data[2] = int(feat_dict['mode'])
            feature_mask[2] = True
            
        # --- Process Numerical ---
        for i, key in enumerate(NUMERICAL_FEATURE_KEYS_ORDERED):
            if key in feat_dict:
                numerical_data[i] = float(feat_dict[key])
                feature_mask[i + 3] = True # +3 to offset the categorical features
                
        return numerical_data, categorical_data, feature_mask


# --- V4.0 Collate Function (Unchanged) ---
def v4_collate_fn_v2(batch: List[Tuple[Tuple, Tuple, Tuple]]) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Custom collate function for the refactored V4.0 Dataset.
    """
    try:
        anchor_batch, positive_batch, negative_batch = zip(*batch)
        
        num_a, cat_a, mask_a = zip(*anchor_batch)
        anchor_tensors = (
            torch.stack(num_a),    # (B, 9)
            torch.stack(cat_a),    # (B, 3)
            torch.stack(mask_a)    # (B, 12)
        )
        
        num_p, cat_p, mask_p = zip(*positive_batch)
        positive_tensors = (
            torch.stack(num_p),
            torch.stack(cat_p),
            torch.stack(mask_p)
        )
        
        num_n, cat_n, mask_n = zip(*negative_batch)
        negative_tensors = (
            torch.stack(num_n),
            torch.stack(cat_n),
            torch.stack(mask_n)
        )
        
        return anchor_tensors, positive_tensors, negative_tensors
    
    except Exception as e:
        logger.error(f"Error in v4_collate_fn_v2: {e}. Batch may be malformed.")
        dummy_tensor = (torch.empty(0, 9), torch.empty(0, 3), torch.empty(0, 12))
        return (dummy_tensor, dummy_tensor, dummy_tensor)