# ml/data_pipeline/specificity_generator_v4.py

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os
import sys
import re # For parsing table values

# Add project root to path to import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logging import logger

# --- V4.0 Configuration ---
OUTPUT_FILE = 'ml/data/synthetic_specificity_v4.parquet' # V4 Parquet output
logger.info(f"V4.0 Specificity Generator. Output will be saved to {OUTPUT_FILE}")

# --- Scale ---
NUM_TRIPLETS_PER_FEATURE_CASE = 9000
NUM_TRIPLETS_FOR_VIBES = 200000
NUM_TRIPLETS_FOR_GENRES = 200000

# Define our 11-feature set
CATEGORICAL_FEATURES = ['key', 'mode']
NUMERICAL_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'speechiness', 'valence', 'tempo', 'loudness'
]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES # 11 features total

# --- Consistent Genre Label Mapping (Matches fma_processor) ---
# Based on the 15 TOP_LEVEL_GENRES used in fma_processor for consistency
TOP_LEVEL_GENRES_NAMES = [
    'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
    'Instrumental', 'International', 'Pop', 'Rock',
    'Jazz', 'Classical', 'Old-Time / Historic', 'Spoken',
    'Blues', 'Soul-RnB', 'Easy Listening'
]
GENRE_TO_LABEL_ID = {name: i for i, name in enumerate(TOP_LEVEL_GENRES_NAMES)}
# *** V4.0 NEW (Task 1.C) ***
LABEL_ID_TO_GENRE = {i: name for name, i in GENRE_TO_LABEL_ID.items()}
DEFAULT_GENRE_NAME = 'Rock'
DEFAULT_GENRE_ID = GENRE_TO_LABEL_ID.get(DEFAULT_GENRE_NAME, 7) 

# --- Helper Functions (Normalization, Parsing) ---
def parse_mu_sigma(val_str):
    """Parses (mean, std_dev) strings from the research doc tables."""
    if pd.isna(val_str) or not isinstance(val_str, str):
        return (0.5, 0.1)
    match = re.match(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', val_str)
    if match:
        mu, sigma = map(float, match.groups())
        return (mu, max(0.0, sigma))
    else:
        try:
            mu = float(val_str)
            return (mu, 0.1) # Assume default std dev if only mean is given
        except ValueError:
            return (0.5, 0.1)

def normalize_loudness(mu_db, sigma_db):
    """Normalizes loudness from dB to 0-1 range."""
    mu_norm = np.clip((mu_db + 60) / 60, 0.0, 1.0)
    sigma_norm = np.clip(sigma_db / 60, 0.0, 0.5) 
    return (mu_norm, sigma_norm)

def normalize_tempo(mu_bpm, sigma_bpm):
    """Normalizes tempo from BPM to 0-1 range."""
    mu_norm = np.clip(mu_bpm / 250.0, 0.0, 1.0)
    sigma_norm = np.clip(sigma_bpm / 250.0, 0.0, 0.5)
    return (mu_norm, sigma_norm)


# --- Vibe Profile Definitions (Directly from Research Doc Table 2) ---
# (Using normalized values)
VIBE_PROFILES = {
    'Positive': {
        'Joyful / Happy': {'danceability': (0.68, 0.15), 'energy': (0.70, 0.18), 'loudness': normalize_loudness(-6.5, 3.5), 'speechiness': (0.07, 0.06), 'acousticness': (0.22, 0.25), 'instrumentalness': (0.02, 0.08), 'liveness': (0.18, 0.14), 'valence': (0.75, 0.15), 'tempo': normalize_tempo(125.0, 26.0), 'key': (5.3, 3.5), 'mode': (0.8, 0.4)},
        'Cheerful': {'danceability': (0.72, 0.14), 'energy': (0.75, 0.16), 'loudness': normalize_loudness(-5.8, 3.1), 'speechiness': (0.08, 0.07), 'acousticness': (0.18, 0.22), 'instrumentalness': (0.01, 0.05), 'liveness': (0.17, 0.13), 'valence': (0.80, 0.12), 'tempo': normalize_tempo(122.0, 24.5), 'key': (5.4, 3.4), 'mode': (0.9, 0.3)},
        'Beautiful': {'danceability': (0.40, 0.18), 'energy': (0.25, 0.20), 'loudness': normalize_loudness(-15.0, 6.0), 'speechiness': (0.04, 0.02), 'acousticness': (0.85, 0.15), 'instrumentalness': (0.60, 0.30), 'liveness': (0.11, 0.09), 'valence': (0.40, 0.20), 'tempo': normalize_tempo(95.0, 30.0), 'key': (5.0, 3.6), 'mode': (0.8, 0.4)},
        'Relaxing': {'danceability': (0.45, 0.20), 'energy': (0.30, 0.22), 'loudness': normalize_loudness(-14.0, 5.5), 'speechiness': (0.05, 0.03), 'acousticness': (0.75, 0.20), 'instrumentalness': (0.70, 0.28), 'liveness': (0.12, 0.10), 'valence': (0.55, 0.22), 'tempo': normalize_tempo(100.0, 28.0), 'key': (5.1, 3.5), 'mode': (0.9, 0.3)},
        'Triumphant': {'danceability': (0.55, 0.16), 'energy': (0.85, 0.12), 'loudness': normalize_loudness(-5.0, 2.8), 'speechiness': (0.06, 0.05), 'acousticness': (0.10, 0.15), 'instrumentalness': (0.25, 0.28), 'liveness': (0.25, 0.18), 'valence': (0.65, 0.20), 'tempo': normalize_tempo(130.0, 29.0), 'key': (5.5, 3.6), 'mode': (0.8, 0.4)},
    },
    'Energetic': {
        'Pumped Up': {'danceability': (0.65, 0.17), 'energy': (0.90, 0.10), 'loudness': normalize_loudness(-4.5, 2.5), 'speechiness': (0.10, 0.08), 'acousticness': (0.05, 0.10), 'instrumentalness': (0.08, 0.18), 'liveness': (0.28, 0.19), 'valence': (0.60, 0.22), 'tempo': normalize_tempo(135.0, 25.0), 'key': (5.2, 3.7), 'mode': (0.7, 0.5)},
        'Defiant': {'danceability': (0.50, 0.18), 'energy': (0.88, 0.11), 'loudness': normalize_loudness(-5.2, 2.7), 'speechiness': (0.12, 0.09), 'acousticness': (0.03, 0.07), 'instrumentalness': (0.10, 0.20), 'liveness': (0.30, 0.20), 'valence': (0.35, 0.20), 'tempo': normalize_tempo(140.0, 30.0), 'key': (4.9, 3.8), 'mode': (0.6, 0.5)},
        'Heroic': {'danceability': (0.52, 0.16), 'energy': (0.92, 0.09), 'loudness': normalize_loudness(-4.8, 2.6), 'speechiness': (0.07, 0.05), 'acousticness': (0.08, 0.12), 'instrumentalness': (0.40, 0.30), 'liveness': (0.26, 0.18), 'valence': (0.50, 0.23), 'tempo': normalize_tempo(132.0, 28.0), 'key': (5.6, 3.5), 'mode': (0.9, 0.3)},
    },
    'Reflective': {
        'Sad': {'danceability': (0.48, 0.18), 'energy': (0.35, 0.20), 'loudness': normalize_loudness(-11.0, 4.5), 'speechiness': (0.05, 0.04), 'acousticness': (0.70, 0.25), 'instrumentalness': (0.10, 0.20), 'liveness': (0.13, 0.11), 'valence': (0.20, 0.15), 'tempo': normalize_tempo(110.0, 32.0), 'key': (4.8, 3.6), 'mode': (0.2, 0.4)},
        'Melancholic': {'danceability': (0.42, 0.17), 'energy': (0.28, 0.18), 'loudness': normalize_loudness(-13.5, 5.0), 'speechiness': (0.04, 0.03), 'acousticness': (0.80, 0.18), 'instrumentalness': (0.25, 0.28), 'liveness': (0.12, 0.10), 'valence': (0.15, 0.12), 'tempo': normalize_tempo(105.0, 35.0), 'key': (4.7, 3.7), 'mode': (0.1, 0.3)},
        'Dreamy': {'danceability': (0.40, 0.20), 'energy': (0.32, 0.22), 'loudness': normalize_loudness(-16.0, 6.2), 'speechiness': (0.04, 0.02), 'acousticness': (0.78, 0.20), 'instrumentalness': (0.80, 0.22), 'liveness': (0.11, 0.09), 'valence': (0.30, 0.20), 'tempo': normalize_tempo(98.0, 33.0), 'key': (5.0, 3.6), 'mode': (0.7, 0.5)},
        'Serene': {'danceability': (0.35, 0.19), 'energy': (0.20, 0.18), 'loudness': normalize_loudness(-18.0, 6.5), 'speechiness': (0.04, 0.02), 'acousticness': (0.90, 0.12), 'instrumentalness': (0.85, 0.20), 'liveness': (0.10, 0.08), 'valence': (0.45, 0.23), 'tempo': normalize_tempo(90.0, 30.0), 'key': (5.2, 3.5), 'mode': (0.9, 0.3)},
    },
    'Intense': {
        'Scary / Anxious': {'danceability': (0.40, 0.22), 'energy': (0.75, 0.20), 'loudness': normalize_loudness(-9.0, 4.8), 'speechiness': (0.08, 0.07), 'acousticness': (0.30, 0.30), 'instrumentalness': (0.70, 0.28), 'liveness': (0.20, 0.17), 'valence': (0.10, 0.10), 'tempo': normalize_tempo(140.0, 40.0), 'key': (4.5, 3.8), 'mode': (0.3, 0.5)},
    }
}
VIBE_RELATIONSHIPS = {
    ('Positive', 'Joyful / Happy'): [('Positive', 'Cheerful'), ('Reflective', 'Sad')],
    ('Positive', 'Cheerful'): [('Positive', 'Joyful / Happy'), ('Reflective', 'Melancholic')],
    ('Positive', 'Beautiful'): [('Positive', 'Relaxing'), ('Energetic', 'Defiant')],
    ('Positive', 'Relaxing'): [('Reflective', 'Serene'), ('Energetic', 'Pumped Up')],
    ('Positive', 'Triumphant'): [('Energetic', 'Heroic'), ('Reflective', 'Melancholic')],
    ('Energetic', 'Pumped Up'): [('Energetic', 'Defiant'), ('Reflective', 'Serene')],
    ('Energetic', 'Defiant'): [('Intense', 'Scary / Anxious'), ('Positive', 'Relaxing')],
    ('Energetic', 'Heroic'): [('Positive', 'Triumphant'), ('Reflective', 'Sad')],
    ('Reflective', 'Sad'): [('Reflective', 'Melancholic'), ('Positive', 'Joyful / Happy')],
    ('Reflective', 'Melancholic'): [('Reflective', 'Sad'), ('Positive', 'Cheerful')],
    ('Reflective', 'Dreamy'): [('Reflective', 'Serene'), ('Energetic', 'Pumped Up')],
    ('Reflective', 'Serene'): [('Reflective', 'Dreamy'), ('Energetic', 'Defiant')],
    ('Intense', 'Scary / Anxious'): [('Energetic', 'Defiant'), ('Positive', 'Beautiful')]
}


# --- Genre Profile Definitions (Directly from Research Doc Table 1) ---
#
GENRE_PROFILES = {
    'Rock': {
        'Overall': {'danceability': (0.55, 0.16), 'energy': (0.72, 0.19), 'loudness': normalize_loudness(-7.5, 3.8), 'speechiness': (0.06, 0.05), 'acousticness': (0.20, 0.25), 'instrumentalness': (0.15, 0.28), 'liveness': (0.20, 0.16), 'valence': (0.51, 0.24), 'tempo': normalize_tempo(124.5, 28.1), 'key': (5.1, 3.6), 'mode': (0.8, 0.4)},
        'Classic Rock': {'danceability': (0.52, 0.14), 'energy': (0.68, 0.18), 'loudness': normalize_loudness(-8.9, 3.5), 'speechiness': (0.04, 0.03), 'acousticness': (0.28, 0.27), 'instrumentalness': (0.08, 0.20), 'liveness': (0.21, 0.18), 'valence': (0.60, 0.22), 'tempo': normalize_tempo(122.1, 25.5), 'key': (5.3, 3.5), 'mode': (0.9, 0.3)},
        'Hard Rock': {'danceability': (0.48, 0.15), 'energy': (0.85, 0.12), 'loudness': normalize_loudness(-5.5, 2.9), 'speechiness': (0.08, 0.06), 'acousticness': (0.05, 0.10), 'instrumentalness': (0.12, 0.25), 'liveness': (0.25, 0.19), 'valence': (0.45, 0.23), 'tempo': normalize_tempo(130.8, 29.3), 'key': (4.9, 3.7), 'mode': (0.7, 0.5)},
        'Indie Rock': {'danceability': (0.58, 0.17), 'energy': (0.65, 0.20), 'loudness': normalize_loudness(-8.0, 4.1), 'speechiness': (0.05, 0.04), 'acousticness': (0.25, 0.28), 'instrumentalness': (0.20, 0.30), 'liveness': (0.18, 0.15), 'valence': (0.48, 0.25), 'tempo': normalize_tempo(125.3, 27.8), 'key': (5.0, 3.6), 'mode': (0.8, 0.4)},
        'Punk': {'danceability': (0.45, 0.18), 'energy': (0.92, 0.08), 'loudness': normalize_loudness(-4.8, 2.5), 'speechiness': (0.10, 0.08), 'acousticness': (0.02, 0.05), 'instrumentalness': (0.05, 0.15), 'liveness': (0.28, 0.20), 'valence': (0.55, 0.26), 'tempo': normalize_tempo(145.0, 35.2), 'key': (5.5, 3.8), 'mode': (0.9, 0.3)},
    },
    'Pop': {
        'Overall': {'danceability': (0.68, 0.14), 'energy': (0.65, 0.18), 'loudness': normalize_loudness(-6.2, 3.1), 'speechiness': (0.08, 0.07), 'acousticness': (0.25, 0.26), 'instrumentalness': (0.01, 0.05), 'liveness': (0.17, 0.14), 'valence': (0.53, 0.23), 'tempo': normalize_tempo(120.1, 25.0), 'key': (5.2, 3.5), 'mode': (0.8, 0.4)},
        'Dance Pop': {'danceability': (0.75, 0.12), 'energy': (0.70, 0.15), 'loudness': normalize_loudness(-5.8, 2.8), 'speechiness': (0.07, 0.06), 'acousticness': (0.15, 0.20), 'instrumentalness': (0.00, 0.02), 'liveness': (0.15, 0.12), 'valence': (0.58, 0.21), 'tempo': normalize_tempo(122.5, 22.1), 'key': (5.4, 3.4), 'mode': (0.7, 0.5)},
        'Electropop': {'danceability': (0.65, 0.15), 'energy': (0.75, 0.17), 'loudness': normalize_loudness(-5.5, 3.0), 'speechiness': (0.06, 0.05), 'acousticness': (0.10, 0.15), 'instrumentalness': (0.05, 0.12), 'liveness': (0.18, 0.15), 'valence': (0.45, 0.24), 'tempo': normalize_tempo(125.0, 26.3), 'key': (5.1, 3.6), 'mode': (0.6, 0.5)},
        'Indie Pop': {'danceability': (0.62, 0.16), 'energy': (0.58, 0.20), 'loudness': normalize_loudness(-7.5, 3.5), 'speechiness': (0.05, 0.04), 'acousticness': (0.35, 0.28), 'instrumentalness': (0.08, 0.18), 'liveness': (0.16, 0.13), 'valence': (0.50, 0.25), 'tempo': normalize_tempo(118.4, 24.8), 'key': (5.3, 3.5), 'mode': (0.9, 0.3)},
    },
    'Hip-Hop': {
        'Overall': {'danceability': (0.75, 0.13), 'energy': (0.62, 0.17), 'loudness': normalize_loudness(-7.8, 3.5), 'speechiness': (0.25, 0.12), 'acousticness': (0.22, 0.24), 'instrumentalness': (0.02, 0.08), 'liveness': (0.18, 0.15), 'valence': (0.45, 0.22), 'tempo': normalize_tempo(125.5, 30.1), 'key': (5.0, 3.7), 'mode': (0.7, 0.5)},
        'Trap': {'danceability': (0.80, 0.11), 'energy': (0.58, 0.15), 'loudness': normalize_loudness(-8.2, 3.2), 'speechiness': (0.28, 0.10), 'acousticness': (0.15, 0.20), 'instrumentalness': (0.01, 0.05), 'liveness': (0.15, 0.13), 'valence': (0.38, 0.20), 'tempo': normalize_tempo(135.2, 28.5), 'key': (4.8, 3.8), 'mode': (0.6, 0.5)},
        'Gangster Rap': {'danceability': (0.72, 0.14), 'energy': (0.68, 0.16), 'loudness': normalize_loudness(-7.0, 3.8), 'speechiness': (0.30, 0.13), 'acousticness': (0.18, 0.22), 'instrumentalness': (0.00, 0.03), 'liveness': (0.22, 0.17), 'valence': (0.42, 0.23), 'tempo': normalize_tempo(95.0, 20.4), 'key': (5.2, 3.6), 'mode': (0.8, 0.4)},
    },
    'Electronic': {
        'Overall': {'danceability': (0.65, 0.18), 'energy': (0.78, 0.18), 'loudness': normalize_loudness(-8.5, 5.0), 'speechiness': (0.07, 0.06), 'acousticness': (0.10, 0.18), 'instrumentalness': (0.55, 0.35), 'liveness': (0.19, 0.16), 'valence': (0.35, 0.25), 'tempo': normalize_tempo(128.0, 20.5), 'key': (5.4, 3.5), 'mode': (0.6, 0.5)},
        'House': {'danceability': (0.72, 0.15), 'energy': (0.80, 0.15), 'loudness': normalize_loudness(-7.5, 4.5), 'speechiness': (0.06, 0.04), 'acousticness': (0.05, 0.10), 'instrumentalness': (0.60, 0.30), 'liveness': (0.17, 0.14), 'valence': (0.40, 0.24), 'tempo': normalize_tempo(125.0, 15.1), 'key': (5.5, 3.4), 'mode': (0.7, 0.5)},
        'Techno': {'danceability': (0.60, 0.19), 'energy': (0.88, 0.12), 'loudness': normalize_loudness(-9.0, 5.2), 'speechiness': (0.08, 0.07), 'acousticness': (0.02, 0.06), 'instrumentalness': (0.75, 0.25), 'liveness': (0.22, 0.18), 'valence': (0.25, 0.22), 'tempo': normalize_tempo(135.0, 18.3), 'key': (5.3, 3.6), 'mode': (0.5, 0.5)},
        'Ambient': {'danceability': (0.30, 0.20), 'energy': (0.25, 0.20), 'loudness': normalize_loudness(-20.0, 7.0), 'speechiness': (0.04, 0.02), 'acousticness': (0.85, 0.15), 'instrumentalness': (0.88, 0.18), 'liveness': (0.10, 0.08), 'valence': (0.15, 0.15), 'tempo': normalize_tempo(90.0, 30.0), 'key': (4.9, 3.7), 'mode': (0.8, 0.4)},
    },
    'Jazz': {
        'Overall': {'danceability': (0.52, 0.17), 'energy': (0.45, 0.25), 'loudness': normalize_loudness(-12.5, 5.5), 'speechiness': (0.06, 0.05), 'acousticness': (0.70, 0.25), 'instrumentalness': (0.65, 0.32), 'liveness': (0.15, 0.13), 'valence': (0.48, 0.26), 'tempo': normalize_tempo(115.3, 32.1), 'key': (5.0, 3.6), 'mode': (0.7, 0.5)},
    },
    'Classical': {
        'Overall': {'danceability': (0.35, 0.15), 'energy': (0.20, 0.18), 'loudness': normalize_loudness(-18.0, 6.8), 'speechiness': (0.05, 0.03), 'acousticness': (0.95, 0.08), 'instrumentalness': (0.80, 0.25), 'liveness': (0.12, 0.10), 'valence': (0.25, 0.20), 'tempo': normalize_tempo(100.2, 35.4), 'key': (5.2, 3.5), 'mode': (0.8, 0.4)},
    },
    'Folk': {
        'Overall': {'danceability': (0.5, 0.15), 'energy': (0.4, 0.2), 'loudness': normalize_loudness(-10.0, 5.0), 'speechiness': (0.05, 0.03), 'acousticness': (0.75, 0.2), 'instrumentalness': (0.1, 0.2), 'liveness': (0.15, 0.1), 'valence': (0.5, 0.2), 'tempo': normalize_tempo(110.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.8, 0.4)},
    },
    'Blues': {
        'Overall': {'danceability': (0.6, 0.15), 'energy': (0.5, 0.2), 'loudness': normalize_loudness(-9.5, 4.0), 'speechiness': (0.07, 0.05), 'acousticness': (0.5, 0.2), 'instrumentalness': (0.2, 0.2), 'liveness': (0.2, 0.1), 'valence': (0.6, 0.2), 'tempo': normalize_tempo(118.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.7, 0.4)},
     },
    'Soul-RnB': {
        'Overall': {'danceability': (0.65, 0.15), 'energy': (0.55, 0.2), 'loudness': normalize_loudness(-8.0, 4.0), 'speechiness': (0.1, 0.07), 'acousticness': (0.3, 0.2), 'instrumentalness': (0.05, 0.1), 'liveness': (0.18, 0.1), 'valence': (0.6, 0.2), 'tempo': normalize_tempo(110.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.7, 0.4)},
    },
    'Experimental': {
        'Overall': {'danceability': (0.4, 0.2), 'energy': (0.6, 0.25), 'loudness': normalize_loudness(-12.0, 6.0), 'speechiness': (0.1, 0.1), 'acousticness': (0.5, 0.3), 'instrumentalness': (0.7, 0.25), 'liveness': (0.25, 0.15), 'valence': (0.2, 0.2), 'tempo': normalize_tempo(120.0, 35.0), 'key': (5.0, 3.5), 'mode': (0.6, 0.5)},
    },
     'Instrumental': {
        'Overall': {'danceability': (0.4, 0.15), 'energy': (0.3, 0.2), 'loudness': normalize_loudness(-16.0, 6.0), 'speechiness': (0.04, 0.02), 'acousticness': (0.8, 0.15), 'instrumentalness': (0.85, 0.15), 'liveness': (0.15, 0.1), 'valence': (0.3, 0.2), 'tempo': normalize_tempo(105.0, 30.0), 'key': (5.0, 3.5), 'mode': (0.8, 0.4)},
    },
    'Old-Time / Historic': {
        'Overall': {'danceability': (0.5, 0.15), 'energy': (0.3, 0.2), 'loudness': normalize_loudness(-12.0, 5.0), 'speechiness': (0.06, 0.04), 'acousticness': (0.85, 0.15), 'instrumentalness': (0.2, 0.2), 'liveness': (0.18, 0.1), 'valence': (0.5, 0.2), 'tempo': normalize_tempo(100.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.8, 0.4)},
    },
    'Easy Listening': { # Added based on adjacency map
        'Overall': {'danceability': (0.5, 0.15), 'energy': (0.3, 0.2), 'loudness': normalize_loudness(-12.0, 5.0), 'speechiness': (0.06, 0.04), 'acousticness': (0.85, 0.15), 'instrumentalness': (0.2, 0.2), 'liveness': (0.18, 0.1), 'valence': (0.5, 0.2), 'tempo': normalize_tempo(100.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.8, 0.4)},
    },
    'Spoken': { # Added based on adjacency map
        'Overall': {'danceability': (0.5, 0.15), 'energy': (0.3, 0.2), 'loudness': normalize_loudness(-12.0, 5.0), 'speechiness': (0.06, 0.04), 'acousticness': (0.85, 0.15), 'instrumentalness': (0.2, 0.2), 'liveness': (0.18, 0.1), 'valence': (0.5, 0.2), 'tempo': normalize_tempo(100.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.8, 0.4)},
    },
    'International': { # Added based on adjacency map
        'Overall': {'danceability': (0.5, 0.15), 'energy': (0.3, 0.2), 'loudness': normalize_loudness(-12.0, 5.0), 'speechiness': (0.06, 0.04), 'acousticness': (0.85, 0.15), 'instrumentalness': (0.2, 0.2), 'liveness': (0.18, 0.1), 'valence': (0.5, 0.2), 'tempo': normalize_tempo(100.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.8, 0.4)},
    },
}
GENRE_RELATIONSHIPS = {
    ('Rock', 'Classic Rock'): [('Rock', 'Hard Rock'), ('Pop', 'Dance Pop')],
    ('Rock', 'Hard Rock'): [('Rock', 'Punk'), ('Electronic', 'Ambient')],
    ('Rock', 'Indie Rock'): [('Pop', 'Indie Pop'), ('Hip-Hop', 'Gangster Rap')],
    ('Rock', 'Punk'): [('Rock', 'Hard Rock'), ('Classical', 'Overall')],
    ('Pop', 'Dance Pop'): [('Pop', 'Electropop'), ('Rock', 'Punk')],
    ('Pop', 'Electropop'): [('Electronic', 'House'), ('Folk', 'Overall')],
    ('Pop', 'Indie Pop'): [('Rock', 'Indie Rock'), ('Hip-Hop', 'Trap')],
    ('Hip-Hop', 'Trap'): [('Hip-Hop', 'Gangster Rap'), ('Classical', 'Overall')],
    ('Hip-Hop', 'Gangster Rap'): [('Hip-Hop', 'Trap'), ('Electronic', 'Ambient')],
    ('Electronic', 'House'): [('Pop', 'Dance Pop'), ('Folk', 'Overall')],
    ('Electronic', 'Techno'): [('Electronic', 'House'), ('Jazz', 'Overall')],
    ('Electronic', 'Ambient'): [('Classical', 'Overall'), ('Rock', 'Punk')],
    ('Rock', 'Overall'): [('Pop', 'Overall'), ('Folk', 'Overall')], 
    ('Pop', 'Overall'): [('Rock', 'Overall'), ('Soul-RnB', 'Overall')],
    ('Hip-Hop', 'Overall'): [('Electronic', 'Overall'), ('Jazz', 'Overall')],
    ('Electronic', 'Overall'): [('Pop', 'Overall'), ('Experimental', 'Overall')],
    ('Jazz', 'Overall'): [('Hip-Hop', 'Overall'), ('Blues', 'Overall')],
    ('Classical', 'Overall'): [('Jazz', 'Overall'), ('Instrumental', 'Overall')],
    ('Folk', 'Overall'): [('Rock', 'Overall'), ('Old-Time / Historic', 'Overall')],
    ('Blues', 'Overall'): [('Rock', 'Overall'), ('Jazz', 'Overall')],
    ('Soul-RnB', 'Overall'): [('Pop', 'Overall'), ('Hip-Hop', 'Overall')],
    ('Experimental', 'Overall'): [('Electronic', 'Overall'), ('Instrumental', 'Overall')],
    ('Instrumental', 'Overall'): [('Classical', 'Overall'), ('Jazz', 'Overall')],
    ('Old-Time / Historic', 'Overall'): [('Folk', 'Overall'), ('Blues', 'Overall')],
    ('Easy Listening', 'Overall'): [('Pop', 'Overall'), ('Jazz', 'Overall')],
    ('Spoken', 'Overall'): [('Experimental', 'Overall'), ('Hip-Hop', 'Overall')],
    ('International', 'Overall'): [('Pop', 'Overall'), ('Folk', 'Overall')],
}


class SpecificityGenerator:
    """
    Generates a V4.0 synthetic dataset of triplets as dictionaries.
    """
    def __init__(self, output_path, logger):
        self.output_path = output_path
        self.logger = logger
        self.final_triplets = [] # Will store tuples of (anchor_dict, pos_dict, neg_dict)

    def _generate_sample_from_profile(self, profile_dict, profile_key):
        """
        Generates a single 11-feature sample dict from a (Category, SubCategory) key.
        """
        if isinstance(profile_key, tuple) and len(profile_key) == 2:
            key1, key2 = profile_key
            if key1 in profile_dict and key2 in profile_dict[key1]:
                profile = profile_dict[key1][key2]
            else:
                 # Fallback to parent 'Overall' if sub-key is missing
                 profile = profile_dict.get(key1, {}).get('Overall', {})
                 if not profile:
                    return self._create_base_sample() # Absolute fallback
        else:
            return self._create_base_sample() # Fallback for bad key

        sample = {}
        for feature in NUMERICAL_FEATURES:
            if feature in profile:
                mean, std = profile[feature]
                val = np.random.normal(mean, std)
                sample[feature] = np.clip(val, 0.0, 1.0)
            else:
                sample[feature] = np.random.rand() # Default if missing

        for feature in CATEGORICAL_FEATURES:
            if feature in profile:
                mean, std = profile[feature]
                if feature == 'key':
                    val = np.random.normal(mean, std)
                    sample[feature] = float(int(round(val)) % 12)
                elif feature == 'mode':
                    sample[feature] = 1.0 if random.random() < mean else 0.0
            else: # Default if missing
                sample[feature] = float(random.randint(0, 11)) if feature == 'key' else float(random.randint(0, 1))
        return sample

    def _create_base_sample(self):
        """Creates an average, randomized 11-feature sample."""
        base_profile = GENRE_PROFILES.get('Rock', {}).get('Overall', {})
        sample = {}
        for feature in NUMERICAL_FEATURES:
            mean, std = base_profile.get(feature, (0.5, 0.1))
            sample[feature] = np.clip(np.random.normal(mean, std), 0.0, 1.0)
        sample['key'] = float(random.randint(0, 11))
        sample['mode'] = float(random.randint(0, 1))
        return sample

    def _perturb_sample(self, sample, amount=0.01):
        """Slightly perturbs the numerical features of a sample."""
        perturbed = sample.copy()
        for feature in NUMERICAL_FEATURES:
            val = perturbed[feature]
            val += np.random.uniform(-amount, amount)
            perturbed[feature] = np.clip(val, 0.0, 1.0)
        return perturbed

    # *** V4.0 MODIFIED (Task 1.C) ***
    def _format_triplet(self, anchor, positive, negative, genre_label_id):
        """
        Formats the triplet. For V4, this means adding the 'genre' string
        to the anchor dictionary.
        """
        # Look up the string name from the ID
        genre_name = LABEL_ID_TO_GENRE.get(genre_label_id, DEFAULT_GENRE_NAME)
        
        # Add the genre string to the anchor dict.
        # Positive and negative dicts do not get a genre string.
        # This is intentional and will help train the model for partial inputs.
        anchor['genre'] = genre_name 
        
        return (anchor, positive, negative)

    def generate_single_feature_triplets(self, num_per_case):
        self.logger.info(f"Generating single-feature hard cases ({num_per_case} each)...")
        start_count = len(self.final_triplets)

        # Assign a default genre ID for these non-genre-specific tests
        default_genre_label = DEFAULT_GENRE_ID

        self.logger.info("... generating for 'mode' (Major vs Minor)")
        for _ in range(num_per_case):
            base = self._create_base_sample()
            base['mode'] = 1.0 # Major
            anchor = self._perturb_sample(base, 0.01)
            positive = self._perturb_sample(base, 0.01)
            negative = anchor.copy()
            negative['mode'] = 0.0 # Minor
            self.final_triplets.append(self._format_triplet(anchor, positive, negative, default_genre_label))

        self.logger.info("... generating for 'key' (Cyclical)")
        for _ in range(num_per_case // 2):
            base = self._create_base_sample()
            base['key'] = float(random.randint(0, 11))
            anchor = self._perturb_sample(base, 0.01)
            positive = self._perturb_sample(base, 0.01)

            # Hard Negative (Adjacent Key)
            negative_hard = anchor.copy()
            negative_hard['key'] = (anchor['key'] + 1) % 12
            self.final_triplets.append(self._format_triplet(anchor, positive, negative_hard, default_genre_label))
            
            # Easy Negative (Distant Key - Tritone)
            negative_easy = anchor.copy()
            negative_easy['key'] = (anchor['key'] + 6) % 12
            self.final_triplets.append(self._format_triplet(anchor, positive, negative_easy, default_genre_label))

        for feature in NUMERICAL_FEATURES:
            self.logger.info(f"... generating for '{feature}'")
            for _ in range(num_per_case // 2):
                base = self._create_base_sample()
                base_val = np.random.rand()
                base[feature] = base_val

                anchor = self._perturb_sample(base, 0.005)
                anchor[feature] = base_val

                positive = self._perturb_sample(base, 0.005)
                positive[feature] = np.clip(base_val + 0.01, 0.0, 1.0) # Very close

                # Hard Negative (a bit further)
                negative_hard = self._perturb_sample(base, 0.005)
                negative_hard[feature] = np.clip(base_val + 0.1, 0.0, 1.0)
                self.final_triplets.append(self._format_triplet(anchor, positive, negative_hard, default_genre_label))

                # Easy Negative (very far)
                negative_easy = self._perturb_sample(base, 0.005)
                negative_easy[feature] = np.clip(base_val + 0.5, 0.0, 1.0)
                self.final_triplets.append(self._format_triplet(anchor, positive, negative_easy, default_genre_label))

        self.logger.info(f"Generated {len(self.final_triplets) - start_count} single-feature triplets.")

    def _generate_profile_triplets(self, num_total, profile_dict, relationship_dict, profile_type):
        self.logger.info(f"Generating {num_total} {profile_type} profile triplets...")
        start_count = len(self.final_triplets)

        all_profile_keys = []
        if profile_type == 'genre':
            for genre, subgenres in profile_dict.items():
                # Only use genres that are in our V4 genre map
                if genre in GENRE_TO_LABEL_ID:
                    for subgenre in subgenres:
                         all_profile_keys.append((genre, subgenre))
        elif profile_type == 'vibe':
             for category, feelings in profile_dict.items():
                for feeling in feelings:
                    all_profile_keys.append((category, feeling))
        else:
             self.logger.error(f"Unknown profile type: {profile_type}")
             return

        if not all_profile_keys:
            self.logger.error(f"No usable profile keys found for type {profile_type}")
            return

        missing_relationships = 0
        generated_count = 0
        pbar = tqdm(total=num_total)
        while generated_count < num_total:
            anchor_key = random.choice(all_profile_keys)

            # --- V4.0: Get the genre label ID for this anchor ---
            # Use the actual genre name for genre profiles
            # Fallback to 'Rock' for vibe profiles
            anchor_genre_name = anchor_key[0] if profile_type == 'genre' else DEFAULT_GENRE_NAME
            anchor_genre_label_id = GENRE_TO_LABEL_ID.get(anchor_genre_name, DEFAULT_GENRE_ID)

            # Find relationships
            relationship = relationship_dict.get(anchor_key)
            if not relationship:
                # Fallback to parent 'Overall' relationship if subgenre is missing
                parent_key = (anchor_key[0], 'Overall')
                relationship = relationship_dict.get(parent_key)
                if not relationship:
                    missing_relationships += 1
                    continue # Skip if no relationship found
            
            hard_neg_key, easy_neg_key = relationship

            # Generate samples (as dicts)
            anchor = self._generate_sample_from_profile(profile_dict, anchor_key)

            # Select positive (allow sibling positive for genres)
            positive_key = anchor_key
            if profile_type == 'genre' and anchor_key[1] != 'Overall': # Only for subgenres
                parent_genre = anchor_key[0]
                siblings = [sg for sg in profile_dict.get(parent_genre, {}) if sg != anchor_key[1] and sg != 'Overall']
                if random.random() < 0.3 and siblings: # 30% chance to use a sibling as positive
                    sibling_subgenre = random.choice(siblings)
                    positive_key = (parent_genre, sibling_subgenre)
            
            positive = self._generate_sample_from_profile(profile_dict, positive_key)

            # Select negative (70% hard, 30% easy)
            if random.random() < 0.7:
                negative = self._generate_sample_from_profile(profile_dict, hard_neg_key)
            else:
                negative = self._generate_sample_from_profile(profile_dict, easy_neg_key)

            # --- V4.0 MODIFIED (Task 1.C) ---
            # Pass the dicts and the genre_label_id to the formatter
            self.final_triplets.append(
                self._format_triplet(anchor, positive, negative, anchor_genre_label_id)
            )
            generated_count += 1
            pbar.update(1)
            
        pbar.close()

        if missing_relationships > 0:
            self.logger.warning(f"Skipped {missing_relationships} triplets due to missing relationships.")
        self.logger.info(f"Generated {generated_count} {profile_type} triplets.")

    # *** V4.0 MODIFIED (Task 1.C) ***
    def save_to_parquet(self):
        """Saves the final list of triplets to a Parquet file."""
        self.logger.info(f"Saving {len(self.final_triplets)} total V4 triplets to {self.output_path}...")

        if not self.final_triplets:
            self.logger.error("No triplets were generated. Stopping.")
            return

        # self.final_triplets is a list of (anchor_dict, pos_dict, neg_dict)
        output_df = pd.DataFrame(self.final_triplets, columns=['anchor', 'positive', 'negative'])

        # Shuffle
        output_df = output_df.sample(frac=1).reset_index(drop=True)

        # Save to Parquet
        output_df.to_parquet(self.output_path, index=False)
        self.logger.info(f"V4.0 Synthetic dataset saved successfully to {self.output_path}")

# --- Main Execution ---
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    generator = SpecificityGenerator(
        output_path=OUTPUT_FILE,
        logger=logger
    )
    
    # 1. Generate single-feature cases
    generator.generate_single_feature_triplets(num_per_case=NUM_TRIPLETS_PER_FEATURE_CASE)
    
    # 2. Generate vibe-based cases
    generator._generate_profile_triplets(
        num_total=NUM_TRIPLETS_FOR_VIBES,
        profile_dict=VIBE_PROFILES,
        relationship_dict=VIBE_RELATIONSHIPS,
        profile_type="vibe"
    )
    
    # 3. Generate genre-based cases
    generator._generate_profile_triplets(
        num_total=NUM_TRIPLETS_FOR_GENRES,
        profile_dict=GENRE_PROFILES,
        relationship_dict=GENRE_RELATIONSHIPS,
        profile_type="genre"
    )

    # 4. Save the final file
    generator.save_to_parquet()

if __name__ == "__main__":
    main()