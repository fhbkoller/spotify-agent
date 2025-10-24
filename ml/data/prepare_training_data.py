import os
import sys
import spotipy
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from urllib.parse import urlparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from utils.logging import setup_logger, logger

# --- Configuration ---
OUTPUT_FILE = "data/training_data.csv"
ANCHOR_PLAYLIST_ID = "4tszLL7NTfLCoIz39Zsiy1"
NEGATIVE_PLAYLIST_ID = "1gKUHcvYNBSesd5HWDlEr4"

# --- Feature Configuration ---
NUMERICAL_COLS = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 
    'liveness', 'loudness', 'speechiness', 'valence', 'tempo'
]
CATEGORICAL_COLS = ['key', 'mode']
ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# --- NEW: Mappings for Categorical Features ---
# We will use these to convert categories to human-readable text for the prompt
KEY_MAP = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}
MODE_MAP = {0: 'Minor', 1: 'Major'}


# --- ReccoBeats API Helper ---
def get_reccobeats_features_batch(track_ids: list[str]) -> dict:
    if not track_ids:
        return {}
    try:
        params = {"ids": ",".join(track_ids)}
        response = requests.get("https://api.reccobeats.com/v1/audio-features", params=params)
        response.raise_for_status()
        data = response.json()
        
        feature_map = {}
        if 'content' in data and isinstance(data['content'], list):
            for features in data['content']:
                if 'href' in features and features['href']:
                    try:
                        path = urlparse(features['href']).path
                        spotify_id = os.path.basename(path)
                        if spotify_id:
                            feature_map[spotify_id] = features
                    except Exception as e:
                        logger.warning(f"Could not parse href: {features.get('href')}. Error: {e}")
        return feature_map
    except requests.exceptions.RequestException as e:
        logger.warning(f"ReccoBeats API batch request failed. Error: {e}")
        return {}

# --- Spotify API Helper ---
def get_spotify_client():
    load_dotenv()
    proxies = {
        'http': os.environ.get('HTTP_PROXY'),
        'https:': os.environ.get('HTTPS_PROXY')
    }
    auth_manager = SpotifyOAuth(scope="playlist-read-private", proxies=proxies)
    return spotipy.Spotify(auth_manager=auth_manager, proxies=proxies)

def _get_playlist_data(sp, playlist_id):
    playlist_tracks = []
    try:
        results = sp.playlist(playlist_id)
        tracks_page = results['tracks']
        while True:
            playlist_tracks.extend(tracks_page['items'])
            if not tracks_page['next']:
                break
            tracks_page = sp.next(tracks_page)
    except Exception as e:
        logger.error(f"Could not fetch playlist {playlist_id}. Details: {e}")
        return []

    valid_tracks = [item['track'] for item in playlist_tracks if item.get('track') and item.get('track').get('id')]
    if not valid_tracks: return []

    artist_ids = list(set([t['artists'][0]['id'] for t in valid_tracks if t.get('artists')]))
    artist_genres_map = {}
    for i in range(0, len(artist_ids), 50):
        try:
            artists_info = sp.artists(artist_ids[i:i+50])
            for artist in artists_info['artists']:
                artist_genres_map[artist['id']] = artist['genres']
        except Exception as e:
            logger.warning(f"Could not fetch artist genres. Details: {e}")

    for track in valid_tracks:
        if track.get('artists'):
            artist_id = track['artists'][0]['id']
            track['genres'] = artist_genres_map.get(artist_id, [])

    return valid_tracks

def get_playlist_features(sp, playlist_id, name):
    logger.info(f"Fetching data for {name} Playlist...")
    tracks = _get_playlist_data(sp, playlist_id)
    if not tracks: return pd.DataFrame()

    track_ids = [t['id'] for t in tracks]
    all_features = {}
    
    logger.info(f"Fetching ReccoBeats features for {len(track_ids)} unique tracks...")
    for i in tqdm(range(0, len(track_ids), 40), desc=f"Fetching {name} Features"):
        all_features.update(get_reccobeats_features_batch(track_ids[i:i+40]))

    data = []
    for t in tracks:
        track_data = {
            'id': t['id'], 
            'name': t['name'], 
            'artist_id': t['artists'][0]['id'] if t.get('artists') else None,
            'genres': t.get('genres', [])
        }
        
        features_from_api = all_features.get(t['id'])
        for col in ALL_FEATURE_COLS:
            track_data[col] = features_from_api.get(col) if features_from_api else None
            
        data.append(track_data)
        
    return pd.DataFrame(data)

# --- Data Processing ---
def clean_and_scale_features(df: pd.DataFrame, scaler=None) -> (pd.DataFrame, object):
    df.dropna(subset=ALL_FEATURE_COLS, inplace=True)
    
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=NUMERICAL_COLS, inplace=True)

    if not df.empty:
        if scaler is None:
            scaler = MinMaxScaler()
            df[NUMERICAL_COLS] = scaler.fit_transform(df[NUMERICAL_COLS])
        else:
            df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
            
    return df, scaler

def create_embedding_prompt(track_series: pd.Series) -> str:
    """
    Creates a simplified text prompt for a track.
    This version OMITS track name and genre to force the model
    to learn from the audio features, key, and mode.
    """
    
    # 1. Build numerical features string
    feature_parts = []
    for col in NUMERICAL_COLS:
        if col == 'tempo':
            feature_parts.append(f"{col}={track_series[col]:.2f}")
        else:
            feature_parts.append(f"{col}={track_series[col]:.3f}")
    features_str = ", ".join(feature_parts)
    
    # 2. Get and map categorical features
    key_val = track_series.get('key')
    mode_val = track_series.get('mode')
    
    try:
        key_str = KEY_MAP.get(int(key_val), 'Unknown')
    except (ValueError, TypeError):
        key_str = 'Unknown'
        
    try:
        mode_str = MODE_MAP.get(int(mode_val), 'Unknown')
    except (ValueError, TypeError):
        mode_str = 'Unknown'

    # 3. Combine *only* features into the final prompt
    # NO track name, NO genre.
    return (
        f"Features: {features_str}, key={key_str}, mode={mode_str}."
    )

def main():
    setup_logger()
    sp = get_spotify_client()

    anchor_df = get_playlist_features(sp, ANCHOR_PLAYLIST_ID, "Anchor")
    negative_df = get_playlist_features(sp, NEGATIVE_PLAYLIST_ID, "Negative")

    if anchor_df.empty or negative_df.empty:
        logger.error("One or both playlists could not be processed. Exiting.")
        sys.exit(1)

    logger.info("Cleaning and preprocessing data...")
    anchor_df, scaler = clean_and_scale_features(anchor_df)
    negative_df, _ = clean_and_scale_features(negative_df, scaler)

    if anchor_df.empty or negative_df.empty:
        logger.error("DataFrames are empty after cleaning. Cannot generate triplets. Exiting.")
        sys.exit(1)

    logger.info("Generating training triplets...")
    training_examples = []
    
    anchor_vectors = anchor_df[NUMERICAL_COLS].to_numpy(dtype=np.float64)

    for idx, (row_index, anchor) in tqdm(enumerate(anchor_df.iterrows()), total=len(anchor_df), desc="Generating Triplets"):
        anchor_genres = set(anchor['genres'])
        
        anchor_vector = anchor_vectors[idx].reshape(1, -1)
        
        positive_candidates = anchor_df[anchor_df['id'] != anchor['id']]
        if positive_candidates.empty:
            continue
        
        candidate_vectors = positive_candidates[NUMERICAL_COLS].to_numpy(dtype=np.float64)
        
        distances = cdist(anchor_vector, candidate_vectors, 'euclidean').flatten()
        
        positive_pool = positive_candidates.copy()
        positive_pool['distance'] = distances
        positive_pool = positive_pool[
            positive_pool['genres'].apply(lambda g: bool(anchor_genres.intersection(g)))
        ].sort_values('distance')
        
        if positive_pool.empty:
            continue

        positive = positive_pool.iloc[0]

        if idx % 2 == 0 and len(positive_pool) > 1:
             hard_negative = positive_pool.iloc[-1]
             negative_choice = hard_negative
        else:
             negative_choice = negative_df.sample(1).iloc[0]

        training_examples.append({
            'anchor': create_embedding_prompt(anchor),
            'positive': create_embedding_prompt(positive),
            'negative': create_embedding_prompt(negative_choice)
        })

    training_df = pd.DataFrame(training_examples)
    file_exists = os.path.exists(OUTPUT_FILE)
    
    if file_exists:
        logger.info(f"Appending {len(training_df)} new triplets to {OUTPUT_FILE}...")
        # Append, and do not write the header
        training_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
    else:
        logger.info(f"Creating new file and saving {len(training_df)} triplets to {OUTPUT_FILE}...")
        # Write new file, including the header
        training_df.to_csv(OUTPUT_FILE, index=False, mode='w', header=True)

    total_triplets = len(pd.read_csv(OUTPUT_FILE))
    logger.info(f"Save operation complete. {OUTPUT_FILE} now contains {total_triplets} total triplets.")

if __name__ == "__main__":
    main()