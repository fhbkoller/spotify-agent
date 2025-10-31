import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os
import sys
import re # For parsing table values

# Add src to path to import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logging import logger

# --- Configuration ---
OUTPUT_FILE = 'ml/data/synthetic_specificity_v2.csv' # Outputting V2 data

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
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# --- Consistent Genre Label Mapping (Matches fma_processor) ---
# Based on the 15 TOP_LEVEL_GENRES used in fma_processor for consistency
TOP_LEVEL_GENRES_NAMES = [
    'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
    'Instrumental', 'International', 'Pop', 'Rock',
    'Jazz', 'Classical', 'Old-Time / Historic', 'Spoken',
    'Blues', 'Soul-RnB', 'Easy Listening'
]
GENRE_TO_LABEL_ID = {name: i for i, name in enumerate(TOP_LEVEL_GENRES_NAMES)}
# Use 'Rock' as the default if a specific genre isn't found in the map
DEFAULT_GENRE_ID = GENRE_TO_LABEL_ID.get('Rock', 7) # Default to Rock's ID

# --- Helper Functions (Normalization, Parsing) ---
def parse_mu_sigma(val_str):
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
    # Normalize dB using the range observed in typical music data (approx -60dB to 0dB)
    # Clip to avoid extreme values from affecting normalization too much
    mu_norm = np.clip((mu_db + 60) / 60, 0.0, 1.0)
    # Normalize std dev relative to the range
    sigma_norm = np.clip(sigma_db / 60, 0.0, 0.5) # Limit max normalized std dev
    return (mu_norm, sigma_norm)

def normalize_tempo(mu_bpm, sigma_bpm):
    # Normalize BPM using a reasonable range (e.g., 0-250 BPM)
    mu_norm = np.clip(mu_bpm / 250.0, 0.0, 1.0)
    sigma_norm = np.clip(sigma_bpm / 250.0, 0.0, 0.5) # Limit max normalized std dev
    return (mu_norm, sigma_norm)


# --- Vibe Profile Definitions (Directly from Research Doc Table 2) ---
#
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
# Define semantic relationships for vibe triplets (Based on research doc logic)
# (Anchor Vibe): [(Hard Negative Vibe), (Easy Negative Vibe)]
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
    # Add other top-level genres needed for relationships, using reasonable defaults if stats are missing
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
    'Old-Time / Historic': { # Added based on adjacency map
        'Overall': {'danceability': (0.5, 0.15), 'energy': (0.3, 0.2), 'loudness': normalize_loudness(-12.0, 5.0), 'speechiness': (0.06, 0.04), 'acousticness': (0.85, 0.15), 'instrumentalness': (0.2, 0.2), 'liveness': (0.18, 0.1), 'valence': (0.5, 0.2), 'tempo': normalize_tempo(100.0, 25.0), 'key': (5.0, 3.5), 'mode': (0.8, 0.4)},
    },
}
# Relationships for genres (Based on research doc logic)
# (Anchor Genre, Anchor SubGenre): [(Hard Negative Genre, SubGenre), (Easy Negative Genre, SubGenre)]
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
    # Adding 'Overall' relationships based on adjacency map in research doc
    ('Rock', 'Overall'): [('Pop', 'Overall'), ('Folk', 'Overall')], # Example: Rock -> adjacent Pop (hard), distant Folk (easy)
    ('Pop', 'Overall'): [('Rock', 'Overall'), ('Soul-RnB', 'Overall')],
    ('Hip-Hop', 'Overall'): [('Electronic', 'Overall'), ('Jazz', 'Overall')],
    ('Electronic', 'Overall'): [('Pop', 'Overall'), ('Experimental', 'Overall')],
    ('Jazz', 'Overall'): [('Hip-Hop', 'Overall'), ('Blues', 'Overall')],
    ('Classical', 'Overall'): [('Jazz', 'Overall'), ('Instrumental', 'Overall')],
    ('Folk', 'Overall'): [('Rock', 'Overall'), ('Old-Time / Historic', 'Overall')],
    # Add fallbacks for any other genres needed
    ('Blues', 'Overall'): [('Rock', 'Overall'), ('Jazz', 'Overall')],
    ('Soul-RnB', 'Overall'): [('Pop', 'Overall'), ('Hip-Hop', 'Overall')],
    ('Experimental', 'Overall'): [('Electronic', 'Overall'), ('Instrumental', 'Overall')],
    ('Instrumental', 'Overall'): [('Classical', 'Overall'), ('Jazz', 'Overall')],
    ('Old-Time / Historic', 'Overall'): [('Folk', 'Overall'), ('Blues', 'Overall')],
}


class SpecificityGenerator:
    def __init__(self, output_path, logger):
        self.output_path = output_path
        self.logger = logger
        self.final_triplets = []

    # _generate_sample_from_profile, _create_base_sample, _perturb_sample are unchanged
    def _generate_sample_from_profile(self, profile_dict, profile_key):
        """Generates a single 11-feature sample from a potentially nested profile."""
        if isinstance(profile_key, tuple) and len(profile_key) == 2:
            key1, key2 = profile_key
            if key1 in profile_dict and key2 in profile_dict[key1]:
                profile = profile_dict[key1][key2]
            else:
                 return self._create_base_sample() # Fallback
        else:
            return self._create_base_sample() # Fallback

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

    # *** CHANGED: Modified to accept and return genre_label ***
    def _format_triplet(self, anchor, positive, negative, genre_label):
        """Flattens a triplet into the CSV row format."""
        all_features = {f'anchor_{f}': anchor[f] for f in FEATURE_COLUMNS}
        all_features.update({f'positive_{f}': positive[f] for f in FEATURE_COLUMNS})
        all_features.update({f'negative_{f}': negative[f] for f in FEATURE_COLUMNS})
        # *** NEW: Add the genre label to the dictionary ***
        all_features['anchor_genre_label'] = genre_label
        return all_features

    # generate_single_feature_triplets is updated to pass the default genre label
    def generate_single_feature_triplets(self, num_per_case):
        self.logger.info(f"Generating single-feature hard cases ({num_per_case} each)...")
        start_count = len(self.final_triplets)

        # *** NEW: Assign a default genre for these non-genre-specific tests ***
        default_genre_label = GENRE_TO_LABEL_ID.get('Rock', DEFAULT_GENRE_ID)

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

            negative_easy = anchor.copy()
            negative_easy['key'] = (anchor['key'] + 6) % 12
            self.final_triplets.append(self._format_triplet(anchor, positive, negative_easy, default_genre_label))

            negative_hard = anchor.copy()
            negative_hard['key'] = (anchor['key'] + 1) % 12
            self.final_triplets.append(self._format_triplet(anchor, positive, negative_hard, default_genre_label))

        for feature in NUMERICAL_FEATURES:
            self.logger.info(f"... generating for '{feature}'")
            for _ in range(num_per_case // 2):
                base = self._create_base_sample()
                base_val = np.random.rand()
                base[feature] = base_val

                anchor = self._perturb_sample(base, 0.005)
                anchor[feature] = base_val

                positive = self._perturb_sample(base, 0.005)
                positive[feature] = np.clip(base_val + 0.01, 0.0, 1.0)

                negative_hard = self._perturb_sample(base, 0.005)
                negative_hard[feature] = np.clip(base_val + 0.1, 0.0, 1.0)
                self.final_triplets.append(self._format_triplet(anchor, positive, negative_hard, default_genre_label))

                negative_easy = self._perturb_sample(base, 0.005)
                negative_easy[feature] = np.clip(base_val + 0.5, 0.0, 1.0)
                self.final_triplets.append(self._format_triplet(anchor, positive, negative_easy, default_genre_label))

        self.logger.info(f"Generated {len(self.final_triplets) - start_count} single-feature triplets.")

    # _generate_profile_triplets is updated to pass the correct genre label
    def _generate_profile_triplets(self, num_total, profile_dict, relationship_dict, profile_type):
        self.logger.info(f"Generating {num_total} {profile_type} profile triplets...")
        start_count = len(self.final_triplets)

        all_profile_keys = []
        if profile_type == 'genre':
            for genre, subgenres in profile_dict.items():
                for subgenre in subgenres:
                     # Ensure the parent genre is one we have a label for
                     if genre in GENRE_TO_LABEL_ID:
                        all_profile_keys.append((genre, subgenre))
                     # else: logger.debug(f"Skipping profile key ({genre}, {subgenre}) - Parent genre not in TOP_LEVEL_GENRES_NAMES")

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

            # *** NEW: Get the genre label ID for this anchor ***
            # Use the actual genre name for genre profiles, fallback to 'Rock' for vibe profiles
            anchor_genre_name = anchor_key[0] if profile_type == 'genre' else 'Rock'
            anchor_genre_label = GENRE_TO_LABEL_ID.get(anchor_genre_name, DEFAULT_GENRE_ID)

            # Find relationships
            if anchor_key not in relationship_dict:
                missing_relationships += 1
                # Fallback logic: Use parent genre relationship if subgenre missing, or random if parent missing
                if profile_type == 'genre':
                    parent_key = (anchor_key[0], 'Overall')
                    if parent_key in relationship_dict:
                        hard_neg_key, easy_neg_key = relationship_dict[parent_key]
                    else: # Parent relationship also missing, choose random different genre
                        possible_negs = [k for k in all_profile_keys if k[0] != anchor_key[0] and profile_type == 'genre']
                        if not possible_negs: continue # Skip if no valid negatives
                        hard_neg_key = random.choice(possible_negs)
                        easy_neg_key = random.choice(possible_negs)
                else: # Fallback for vibes (choose random different category)
                     possible_negs = [k for k in all_profile_keys if k[0] != anchor_key[0] and profile_type == 'vibe']
                     if not possible_negs: continue
                     hard_neg_key = random.choice(possible_negs)
                     easy_neg_key = random.choice(possible_negs)
            else:
                 hard_neg_key, easy_neg_key = relationship_dict[anchor_key]

            # Generate samples
            anchor = self._generate_sample_from_profile(profile_dict, anchor_key)

            # Select positive (allow sibling positive for genres)
            positive_key = anchor_key
            if profile_type == 'genre' and anchor_key[1] != 'Overall': # Only for subgenres
                parent_genre = anchor_key[0]
                # Find siblings (other subgenres under the same parent)
                siblings = [sg for sg in profile_dict.get(parent_genre, {}) if sg != anchor_key[1] and sg != 'Overall']
                # 30% chance to use a sibling as positive
                if random.random() < 0.3 and siblings:
                    sibling_subgenre = random.choice(siblings)
                    positive_key = (parent_genre, sibling_subgenre)
            positive = self._generate_sample_from_profile(profile_dict, positive_key)

            # Select negative (70% hard, 30% easy)
            if random.random() < 0.7:
                negative = self._generate_sample_from_profile(profile_dict, hard_neg_key)
            else:
                negative = self._generate_sample_from_profile(profile_dict, easy_neg_key)

            # *** CHANGED: Pass the real genre label ***
            self.final_triplets.append(self._format_triplet(anchor, positive, negative, anchor_genre_label))
            generated_count += 1
            pbar.update(1)
        pbar.close()

        if missing_relationships > 0:
            self.logger.warning(f"Used fallback relationships for {missing_relationships} out of {num_total} {profile_type} triplets.")
        self.logger.info(f"Generated {generated_count} {profile_type} triplets.")


    # save_to_csv is updated to handle the new genre_label column and ensure order
    def save_to_csv(self):
        self.logger.info(f"Saving {len(self.final_triplets)} total triplets to {self.output_path}...")

        if not self.final_triplets:
            self.logger.error("No triplets were generated. Stopping.")
            return

        output_df = pd.DataFrame(self.final_triplets)

        # Rename descriptive feature names to indexed names ('feat_0', 'feat_1', ...)
        rename_map = {}
        for i, feature_name in enumerate(FEATURE_COLUMNS):
            rename_map[f'anchor_{feature_name}'] = f'anchor_feat_{i}'
            rename_map[f'positive_{feature_name}'] = f'positive_feat_{i}'
            rename_map[f'negative_{feature_name}'] = f'negative_feat_{i}'

        output_df.rename(columns=rename_map, inplace=True)

        # *** NEW: Define final column order, starting with genre label ***
        # (anchor_genre_label, anchor_feat_0...10, positive_feat_0...10, negative_feat_0...10)
        final_ordered_cols = ['anchor_genre_label']
        for prefix in ['anchor', 'positive', 'negative']:
            for i in range(len(FEATURE_COLUMNS)):
                final_ordered_cols.append(f'{prefix}_feat_{i}')

        # Ensure all expected columns exist, add if missing (with default value)
        for col in final_ordered_cols:
            if col not in output_df:
                logger.warning(f"Column {col} missing from generated data. Adding with default.")
                # Use -1 for missing label, 0.0 for missing features
                default_value = -1 if col == 'anchor_genre_label' else 0.0
                output_df[col] = default_value

        # Reorder DataFrame columns
        output_df = output_df[final_ordered_cols]

        output_df.to_csv(self.output_path, index=False)
        self.logger.info("Done.")

# --- Main Execution ---
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    generator = SpecificityGenerator(
        output_path=OUTPUT_FILE,
        logger=logger
    )

    generator.generate_single_feature_triplets(num_per_case=NUM_TRIPLETS_PER_FEATURE_CASE)

    generator._generate_profile_triplets(
        num_total=NUM_TRIPLETS_FOR_VIBES,
        profile_dict=VIBE_PROFILES,
        relationship_dict=VIBE_RELATIONSHIPS,
        profile_type="vibe"
    )

    generator._generate_profile_triplets(
        num_total=NUM_TRIPLETS_FOR_GENRES,
        profile_dict=GENRE_PROFILES,
        relationship_dict=GENRE_RELATIONSHIPS,
        profile_type="genre"
    )

    generator.save_to_csv()

if __name__ == "__main__":
    main()