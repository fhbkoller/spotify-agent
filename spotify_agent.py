import os
import time
import json
import ollama
import spotipy
import requests
import numpy as np
import logging
from tqdm import tqdm
from spotipy.oauth2 import SpotifyOAuth
from dataclasses import dataclass, field

# Import the new PersistenceManager
from persistence import PersistenceManager

logger = logging.getLogger(__name__)

@dataclass
class Track:
    """A dataclass to hold the in-memory representation of a track's state."""
    id: str
    name: str
    artist: str
    embedding: np.ndarray = field(repr=False)
    score: float = 1.0
    skip_count: int = 0

class IntelligentShuffler:
    # --- Agent Configuration ---
    EMBEDDING_MODEL = "mxbai-embed-large"
    GENERATIVE_MODEL = "qwen:7b"

    # --- Tuning Parameters ---
    SKIP_PENALTY = 0.5
    FINISH_BONUS = 0.1
    POLLING_INTERVAL_SECONDS = 5

    def __init__(self, playlist_id: str):
        self._setup_spotify_client()
        self.playlist_id = playlist_id
        self.user_id = self.sp.me()["id"]

        # Instantiate the PersistenceManager
        # The database is now the source of truth for all track data.
        self.db = PersistenceManager()

        # In-memory dictionaries are now caches, populated from the database.
        self.playlist_tracks: dict[str, Track] = {}
        self.library_tracks: dict[str, Track] = {}

        # State for the main run loop
        self.current_track_id: str | None = None
        self.current_track_item: dict | None = None
        self.last_progress_ms: int = 0
        self.song_added_for_current_track = False

    def _setup_spotify_client(self):
        scope = (
            "user-read-playback-state user-modify-playback-state "
            "playlist-modify-public playlist-read-private user-library-read"
        )
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
        logger.debug("Spotify client set up successfully.")

    def initialize(self):
        logger.info("Initializing Intelligent Shuffle Agent...")
        logger.info(f"Using database-first approach for persistence.")

        logger.info("Syncing tracks from target playlist...")
        self._load_and_sync_tracks(
            fetcher=self.sp.playlist_items, 
            cache=self.playlist_tracks,
            playlist_id=self.playlist_id
        )
        logger.info(f"Loaded {len(self.playlist_tracks)} tracks for playlist.")

        logger.info("Syncing tracks from user library...")
        self._load_and_sync_tracks(
            fetcher=self.sp.current_user_saved_tracks, 
            cache=self.library_tracks,
            playlist_id=None
        )
        logger.info(f"Loaded {len(self.library_tracks)} tracks from library.")

    def _load_and_sync_tracks(self, fetcher, cache: dict, playlist_id: str | None):
        """
        Loads tracks from a Spotify source (playlist/library), checks against the
        local DB, fetches missing data from APIs, and saves new tracks.
        """
        all_spotify_ids = self._get_all_track_ids_from_fetcher(fetcher, playlist_id)
        if not all_spotify_ids:
            logger.warning("Source contains no tracks.")
            return

        found_tracks_data, missing_ids = self.db.get_tracks_by_ids(all_spotify_ids)

        for track_id, track_data in found_tracks_data.items():
            embedding = track_data.get('embedding')
            if embedding is not None:
                cache[track_id] = Track(
                    id=track_id, name=track_data['name'], artist=track_data['artist'],
                    score=track_data['score'], skip_count=track_data['skip_count'],
                    embedding=embedding
                )
        logger.info(f"Loaded {len(cache)} tracks from local database.")

        if missing_ids:
            logger.info(f"Fetching data for {len(missing_ids)} new tracks...")
            missing_ids_batches = [missing_ids[i:i + 50] for i in range(0, len(missing_ids), 50)]
            
            for id_batch in tqdm(missing_ids_batches, desc="Syncing new tracks"):
                spotify_tracks_info = self.sp.tracks(id_batch)['tracks']
                reccobeats_features = self._get_reccobeats_features_batch(id_batch)

                for track_info in spotify_tracks_info:
                    if not track_info: continue
                    track_id = track_info['id']
                    track_features = reccobeats_features.get(track_id)
                    if not track_features:
                        logger.warning(f"Skipping new track {track_info['name']} due to missing audio features.")
                        continue
                    
                    prompt = self._create_embedding_prompt(track_info, track_features)
                    embedding = self._get_embedding(prompt)
                    
                    new_track_data = {
                        "spotify_id": track_id, "name": track_info['name'],
                        "artist": track_info['artists'][0]['name'], "embedding": embedding,
                    }
                    
                    self.db.save_track(new_track_data)
                    cache[track_id] = Track(
                        id=track_id, name=new_track_data['name'], artist=new_track_data['artist'],
                        embedding=embedding, score=1.0, skip_count=0
                    )

    def _get_all_track_ids_from_fetcher(self, paged_fetcher, playlist_id) -> list[str]:
        items = []
        results = paged_fetcher(playlist_id) if playlist_id else paged_fetcher()
        
        while results:
            items.extend(results["items"])
            results = self.sp.next(results) if results["next"] else None
        
        return [item['track']['id'] for item in items if item.get('track') and item['track'].get('id')]
    
    def run(self):
        logger.info("Starting playback loop...")
        while True:
            try:
                playback = self.sp.current_playback()

                if not playback or not playback.get("is_playing"):
                    if self.current_track_item:
                        duration_ms = self.current_track_item.get('duration_ms', 0)
                        was_song_finished = duration_ms > 0 and self.last_progress_ms > (duration_ms * 0.98)
                        
                        skipped = not was_song_finished
                        logger.info(f"Playback stopped. Evaluating last track: '{self.current_track_item['name']}' (skipped: {skipped})")
                        self._update_scores(skipped=skipped)
                        
                        last_played_track_id = self.current_track_id
                        self.current_track_id = None
                        self.current_track_item = None
                        
                        if was_song_finished and self.playlist_tracks:
                            next_track_id = self._get_next_track(last_played_track_id)
                            if next_track_id:
                                logger.info(f"▶️ Queue ended. Playing next weighted-random track: {self.playlist_tracks[next_track_id].name}")
                                self.sp.start_playback(uris=[f'spotify:track:{next_track_id}'])
                            else:
                                logger.warning("Queue ended but could not determine next track.")
                    
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue

                new_track_item = playback.get("item")
                if not new_track_item:
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue
                
                new_track_id = new_track_item["id"]

                if new_track_id != self.current_track_id:
                    if self.current_track_item:
                        duration_ms = self.current_track_item.get('duration_ms', 0)
                        skipped = duration_ms > 0 and self.last_progress_ms < (duration_ms * 0.85)
                        logger.info(f"New song detected. Evaluating last track: '{self.current_track_item['name']}' (progress: {self.last_progress_ms}ms / {duration_ms}ms, skipped: {skipped})")
                        self._update_scores(skipped=skipped)

                    self.current_track_id = new_track_id
                    self.current_track_item = new_track_item
                    self.last_progress_ms = 0
                    self.song_added_for_current_track = False
                    logger.info(f"🎶 Now Playing: '{new_track_item['name']}' by {new_track_item['artists'][0]['name']}")

                self.last_progress_ms = playback.get('progress_ms', 0)

                duration_ms = self.current_track_item.get('duration_ms', 0)
                if not self.song_added_for_current_track and duration_ms > 0 and self.last_progress_ms > (duration_ms * 0.80):
                    self._add_new_song()
                    self.song_added_for_current_track = True

            except Exception as e:
                logger.error(f"An error occurred in the playback loop: {e}", exc_info=True)
                time.sleep(30)

            time.sleep(self.POLLING_INTERVAL_SECONDS)

    def _update_scores(self, skipped: bool):
        track_id = self.current_track_id
        if not track_id or track_id not in self.playlist_tracks:
            return

        track = self.playlist_tracks[track_id]
        if skipped:
            track.score *= self.SKIP_PENALTY
            track.skip_count += 1
            logger.info(f"🔻 Score for '{track.name}' decreased to {track.score:.2f}")
        else:
            track.score += self.FINISH_BONUS
            logger.info(f"🔺 Score for '{track.name}' increased to {track.score:.2f}")
        
        self.db.update_track_stats(track.id, track.score, track.skip_count)
        
        if track.skip_count >= 3:
            logger.warning(f"Removing '{track.name}' from playlist due to repeated skips.")
            self.sp.remove_playlist_items(self.playlist_id, [track.id])
            del self.playlist_tracks[track.id]

    def _get_next_track(self, last_played_id: str | None = None) -> str:
        if not self.playlist_tracks:
            return None

        candidate_tracks = {tid: track for tid, track in self.playlist_tracks.items() if tid != last_played_id}
        if not candidate_tracks:
            return list(self.playlist_tracks.keys())[0]

        scores = np.array([track.score for track in candidate_tracks.values()])
        probabilities = scores / scores.sum()
        
        track_ids = list(candidate_tracks.keys())
        return np.random.choice(track_ids, p=probabilities)

    def _add_new_song(self):
        logger.debug("Attempting to add a new song.")
        if np.random.rand() > 0.3:
            self._add_similar_song()
        else:
            self._add_generative_song()

    def _add_similar_song(self):
        logger.debug("Attempting to find a similar song from the database.")
        if not self.current_track_id or self.current_track_id not in self.playlist_tracks:
            logger.debug("Aborting similar song search: No current track.")
            return

        current_track = self.playlist_tracks[self.current_track_id]
        
        similar_ids = self.db.find_similar_tracks(
            vector=current_track.embedding, n_results=5,
            exclude_ids=list(self.playlist_tracks.keys())
        )
        
        if not similar_ids:
            logger.info("[Similarity] No suitable new tracks found in the database.")
            return

        best_candidate_id = similar_ids[0]
        
        if best_candidate_id not in self.library_tracks:
             self._load_and_sync_tracks(lambda ids: self.sp.tracks(ids), self.library_tracks, [best_candidate_id])

        best_candidate_track = self.library_tracks.get(best_candidate_id)
        if best_candidate_track:
            logger.info(f"✨ [Similarity] Found a good match: '{best_candidate_track.name}' by {best_candidate_track.artist}")
            self.sp.add_to_playlist(self.playlist_id, [best_candidate_track.id])
            self.playlist_tracks[best_candidate_track.id] = best_candidate_track

    def _get_reccobeats_features_batch(self, spotify_track_ids: list[str]) -> dict:
        if not spotify_track_ids: return {}
        headers = {"Accept": "application/json"}
        params = {"ids": ",".join(spotify_track_ids)}
        try:
            response = requests.get("https://api.reccobeats.com/v1/audio-features", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if 'content' in data and isinstance(data['content'], list):
                if len(spotify_track_ids) != len(data['content']):
                    logger.warning(f"ReccoBeats ID/result mismatch. Sent {len(spotify_track_ids)}, received {len(data['content'])}.")
                    return {}
                return {sid: features for sid, features in zip(spotify_track_ids, data['content'])}
            else:
                logger.warning(f"ReccoBeats API returned unexpected data format: {data}")
                return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get audio features from ReccoBeats: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ReccoBeats JSON response: {e}")
            logger.debug(f"ReccoBeats raw response text: {response.text}")
        return {}

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            response = ollama.embeddings(model=self.EMBEDDING_MODEL, prompt=text)
            return np.array(response["embedding"])
        except Exception as e:
            logger.error(f"Failed to get embedding from Ollama model '{self.EMBEDDING_MODEL}': {e}")
            raise

    def _create_embedding_prompt(self, track_info: dict, reccobeats_data: dict) -> str:
        return f"{track_info['name']} by {track_info['artists'][0]['name']} - Tempo: {reccobeats_data.get('tempo', 'N/A')}, Danceability: {reccobeats_data.get('danceability', 'N/A')}"

    def _add_generative_song(self):
        if not self.playlist_tracks: return
        history_summary = "\n".join([f"- '{t.name}' by {t.artist}" for t in list(self.playlist_tracks.values())[-5:]])
        prompt = (f"Based on these recently played songs:\n{history_summary}\n\n"
                  "Recommend one new song that would be a great addition. Respond with only a JSON object: "
                  '{"song_name": "SONG_NAME", "artist_name": "ARTIST_NAME"}')

        try:
            response = ollama.chat(model=self.GENERATIVE_MODEL, messages=[{'role': 'user', 'content': prompt}], format="json")
            recommendation = json.loads(response['message']['content'])
            song_name, artist_name = recommendation['song_name'], recommendation['artist_name']
        except Exception as e:
            logger.error(f"Error with generative suggestion: {e}")
            return

        logger.info(f"🤖 LLM suggested: '{song_name}' by {artist_name}")
        search_result = self.sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
        
        if not search_result['tracks']['items']:
            logger.warning(f"LLM suggestion '{song_name}' not found on Spotify.")
            return

        new_track_info = search_result['tracks']['items'][0]
        if new_track_info['id'] in self.playlist_tracks:
            logger.info("LLM suggested a song already in the playlist.")
            return

        reccobeats_data = self._get_reccobeats_features_batch([new_track_info['id']]).get(new_track_info['id'])
        if not reccobeats_data:
            logger.warning(f"Could not get ReccoBeats data for LLM suggestion '{song_name}'.")
            return

        embedding = self._get_embedding(self._create_embedding_prompt(new_track_info, reccobeats_data))
        new_track = Track(
            id=new_track_info['id'], name=new_track_info['name'],
            artist=new_track_info['artists'][0]['name'], embedding=embedding
        )
        logger.info(f"➕ [Generative] Adding: '{new_track.name}' by {new_track.artist}")
        self.sp.add_to_playlist(self.playlist_id, [new_track.id])
        self.playlist_tracks[new_track.id] = new_track