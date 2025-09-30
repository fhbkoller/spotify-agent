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

logger = logging.getLogger(__name__)

@dataclass
class Track:
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
        self.playlist_tracks: dict[str, Track] = {}
        self.library_tracks: dict[str, Track] = {}

        # --- State for the main run loop ---
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

    # ... (no changes in most helper methods)
    def _get_reccobeats_features_batch(self, spotify_track_ids: list[str]) -> dict:
        if not spotify_track_ids:
            return {}
        headers = {"Accept": "application/json"}
        params = {"ids": ",".join(spotify_track_ids)}
        try:
            response = requests.get(
                "https://api.reccobeats.com/v1/audio-features",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            if 'content' in data and isinstance(data['content'], list):
                if len(spotify_track_ids) != len(data['content']):
                    logger.warning(
                        f"ReccoBeats ID/result mismatch. "
                        f"Sent {len(spotify_track_ids)} IDs, "
                        f"received {len(data['content'])} results."
                    )
                    return {}
                return {
                    spotify_id: features
                    for spotify_id, features in zip(spotify_track_ids, data['content'])
                }
            else:
                logger.warning(f"ReccoBeats API returned unexpected data format: {data}")
                return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get audio features from ReccoBeats: {e}")
            return {}
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
        return f"{track_info['name']} by {track_info['artists'][0]['name']} - " f"Tempo: {reccobeats_data.get('tempo', 'N/A')}, " f"Danceability: {reccobeats_data.get('danceability', 'N/A')}"

    def initialize(self):
        logger.info("Initializing Intelligent Shuffle Agent...")
        logger.info(f"Using embedding model: {self.EMBEDDING_MODEL}")
        logger.info(f"Using generative model: {self.GENERATIVE_MODEL}")

        logger.info("Fetching tracks from target playlist...")
        self._fetch_and_embed_tracks(
            self.sp.playlist_items, self.playlist_id, self.playlist_tracks
        )
        logger.info(f"Found and embedded {len(self.playlist_tracks)} tracks from playlist.")

        logger.info("Fetching tracks from user library...")
        self._fetch_and_embed_tracks(self.sp.current_user_saved_tracks, None, self.library_tracks)
        logger.info(f"Found and embedded {len(self.library_tracks)} tracks from library.")

    def _fetch_and_embed_tracks(self, paged_fetcher, playlist_id, track_dict):
        # (This function's logic remains the same, no changes needed here)
        if playlist_id:
            results = paged_fetcher(playlist_id)
        else:
            results = paged_fetcher()

        all_track_items = []
        while results:
            all_track_items.extend(results["items"])
            if results["next"]:
                results = self.sp.next(results)
            else:
                results = None
        
        track_ids = [item['track']['id'] for item in all_track_items if item.get('track')]
        track_ids_batches = [track_ids[i:i + 40] for i in range(0, len(track_ids), 40)]

        for track_ids_batch in tqdm(track_ids_batches, desc="Analyzing tracks"):
            reccobeats_features = self._get_reccobeats_features_batch(track_ids_batch)
            
            if not reccobeats_features:
                logger.warning(f"Skipping batch due to missing ReccoBeats data.")
                continue

            for item in all_track_items:
                if not item.get('track') or item['track']['id'] not in track_ids_batch:
                    continue
                
                track_id = item['track']['id']
                track_features = reccobeats_features.get(track_id)
                
                if not track_features:
                    logger.warning(f"No ReccoBeats features for track ID: {track_id}. Skipping embedding.")
                    continue

                embedding_prompt = self._create_embedding_prompt(item['track'], track_features)
                embedding = self._get_embedding(embedding_prompt)
                
                track_dict[track_id] = Track(
                    id=track_id,
                    name=item['track']['name'],
                    artist=item['track']['artists'][0]['name'],
                    embedding=embedding,
                )
    
    def run(self):
        logger.info("Starting playback loop...")
        while True:
            try:
                playback = self.sp.current_playback()

                # Case 1: Playback is paused, stopped, or unavailable.
                if not playback or not playback.get("is_playing"):
                    if self.current_track_item:
                        duration_ms = self.current_track_item.get('duration_ms', 0)
                        
                        # Heuristic: If progress was > 98% of duration, song finished naturally.
                        was_song_finished = duration_ms > 0 and self.last_progress_ms > (duration_ms * 0.98)
                        
                        # A finished song is not a skip.
                        skipped = not was_song_finished
                        logger.info(f"Playback stopped. Evaluating last track: '{self.current_track_item['name']}' (skipped: {skipped})")
                        self._update_scores(skipped=skipped)
                        
                        # Store the previous track ID and item before clearing them
                        last_played_track_id = self.current_track_id
                        self.current_track_id = None
                        self.current_track_item = None
                        
                        # If the song finished, it's time to pick the next one based on scores.
                        if was_song_finished and self.playlist_tracks:
                            next_track_id = self._get_next_track(last_played_track_id)
                            if next_track_id:
                                logger.info(f"▶️ Queue ended. Playing next weighted-random track: {self.playlist_tracks[next_track_id].name}")
                                self.sp.start_playback(uris=[f'spotify:track:{next_track_id}'])
                            else:
                                logger.warning("Queue ended but could not determine next track.")
                    
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue

                # Case 2: Something is playing.
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
        track_to_update_id = self.current_track_id
        if not track_to_update_id or track_to_update_id not in self.playlist_tracks:
            return

        track = self.playlist_tracks[track_to_update_id]
        if skipped:
            track.score *= self.SKIP_PENALTY
            track.skip_count += 1
            logger.info(f"🔻 Score for '{track.name}' decreased to {track.score:.2f}")
        else:
            track.score += self.FINISH_BONUS
            logger.info(f"🔺 Score for '{track.name}' increased to {track.score:.2f}")
        
        if track.skip_count >= 3:
            logger.warning(f"Removing '{track.name}' from playlist due to repeated skips.")
            self.sp.remove_playlist_items(self.playlist_id, [track.id])
            del self.playlist_tracks[track.id]

    def _get_next_track(self, last_played_id: str | None = None) -> str:
        if not self.playlist_tracks:
            return None

        # Filter out the last song played to avoid immediate repetition
        candidate_tracks = {tid: track for tid, track in self.playlist_tracks.items() if tid != last_played_id}
        if not candidate_tracks:
            # If there's only one song in the playlist, just play it again.
            return list(self.playlist_tracks.keys())[0]

        scores = np.array([track.score for track in candidate_tracks.values()])
        probabilities = scores / scores.sum()
        
        track_ids = list(candidate_tracks.keys())
        next_track_id = np.random.choice(track_ids, p=probabilities)
        return next_track_id

    def _add_new_song(self):
        logger.debug("Attempting to add a new song.")
        if np.random.rand() > 0.3:
            self._add_similar_song()
        else:
            self._add_generative_song()

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _add_similar_song(self):
        logger.debug("Attempting to find a similar song from the user's library.")
        if not self.current_track_id or self.current_track_id not in self.playlist_tracks:
            logger.debug("Aborting similar song search: No current track in playlist.")
            return

        if not self.library_tracks:
            logger.warning("Cannot find similar song: User library is empty or failed to load.")
            return

        current_embedding = self.playlist_tracks[self.current_track_id].embedding
        
        best_candidate = None
        max_similarity = -1

        for track_id, track in self.library_tracks.items():
            if track_id in self.playlist_tracks:
                continue
            
            similarity = self._cosine_similarity(current_embedding, track.embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_candidate = track

        if best_candidate:
            logger.info(f"✨ [Similarity] Found a good match: '{best_candidate.name}' by {best_candidate.artist}")
            self.sp.add_to_playlist(self.playlist_id, [best_candidate.id])
            self.playlist_tracks[best_candidate.id] = best_candidate
        else:
            logger.info("[Similarity] No suitable new tracks found in the user's library to add.")
            
    def _add_generative_song(self):
        # ... (no changes in this method)
        if not self.playlist_tracks:
            return

        history_summary = "\n".join([f"- '{t.name}' by {t.artist}" for t in list(self.playlist_tracks.values())[-5:]])
        prompt = (
            "Based on the following recently played songs:\n"
            f"{history_summary}\n\n"
            "Recommend one new song (not from this list) that would be a great addition. "
            "Respond with only a JSON object in the format: "
            '{"song_name": "SONG_NAME", "artist_name": "ARTIST_NAME"}'
        )

        try:
            response = ollama.chat(model=self.GENERATIVE_MODEL, messages=[{'role': 'user', 'content': prompt}], format="json")
        except Exception as e:
            logger.error(f"Error calling Ollama '{self.GENERATIVE_MODEL}': {e}")
            return

        if response and response['message']['content']:
            try:
                recommendation = json.loads(response['message']['content'])
                song_name, artist_name = recommendation['song_name'], recommendation['artist_name']
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse LLM response: {e}")
                logger.debug(f"Raw LLM response: {response['message']['content']}")
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
                logger.warning(f"Could not get ReccoBeats data for LLM suggestion '{song_name}'. Skipping.")
                return

            embedding = self._get_embedding(self._create_embedding_prompt(new_track_info, reccobeats_data))

            new_track = Track(
                id=new_track_info['id'], name=new_track_info['name'],
                artist=new_track_info['artists'][0]['name'], embedding=embedding
            )
            logger.info(f"➕ [Generative] Adding: '{new_track.name}' by {new_track.artist}")
            self.sp.add_to_playlist(self.playlist_id, [new_track.id])
            self.playlist_tracks[new_track.id] = new_track