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

from persistence import PersistenceManager

logger = logging.getLogger(__name__)

### --- FIX: Added the missing 'embedding' field --- ###
@dataclass
class Track:
    id: str
    name: str
    artist: str
    embedding: np.ndarray = field(repr=False)
    score: float = 1.0
    skip_count: int = 0

class IntelligentShuffler:
    EMBEDDING_MODEL = "mxbai-embed-large"
    GENERATIVE_MODEL = "qwen:7b"
    API_BATCH_SIZE = 40
    SKIP_PENALTY = 0.5
    FINISH_BONUS = 0.1
    POLLING_INTERVAL_SECONDS = 5

    def __init__(self, playlist_id: str):
        self._setup_spotify_client()
        self.playlist_id = playlist_id
        self.user_id = self.sp.me()["id"]
        self.db = PersistenceManager()
        self.playlist_tracks: dict[str, Track] = {}
        self.library_tracks: dict[str, Track] = {}
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
        
        logger.info("Syncing tracks from target playlist...")
        playlist_items = []
        results = self.sp.playlist_items(self.playlist_id, limit=100)
        while results:
            playlist_items.extend(results.get("items", []))
            results = self.sp.next(results) if results.get("next") else None
        logger.info(f"Fetched a total of {len(playlist_items)} items from Spotify playlist.")
        playlist_ids = [item['track']['id'] for item in playlist_items if item.get('track') and item['track'].get('id')]
        self.playlist_tracks = self._sync_tracks_with_db(playlist_ids)
        logger.info(f"Loaded {len(self.playlist_tracks)} tracks for playlist.")

        logger.info("Syncing tracks from user library...")
        library_items = []
        results = self.sp.current_user_saved_tracks(limit=50)
        while results:
            library_items.extend(results.get("items", []))
            results = self.sp.next(results) if results.get("next") else None
        logger.info(f"Fetched a total of {len(library_items)} items from Spotify library.")
        library_ids = [item['track']['id'] for item in library_items if item.get('track') and item['track'].get('id')]
        self.library_tracks = self._sync_tracks_with_db(library_ids)
        logger.info(f"Loaded {len(self.library_tracks)} tracks from library.")

    def _sync_tracks_with_db(self, spotify_ids: list[str]) -> dict[str, Track]:
        cache = {}
        if not spotify_ids: return cache

        found_tracks_data, missing_ids = self.db.get_tracks_by_ids(spotify_ids)

        for track_id, track_data in found_tracks_data.items():
            embedding = track_data.get('embedding')
            if embedding is not None:
                cache[track_id] = Track(**track_data)
        if found_tracks_data:
            logger.info(f"Loaded {len(found_tracks_data)} tracks from local database.")

        if missing_ids:
            logger.info(f"Fetching data for {len(missing_ids)} new tracks...")
            batches = [missing_ids[i:i + self.API_BATCH_SIZE] for i in range(0, len(missing_ids), self.API_BATCH_SIZE)]
            for id_batch in tqdm(batches, desc="Syncing new tracks"):
                spotify_tracks = self.sp.tracks(id_batch)['tracks']
                recco_features = self._get_reccobeats_features_batch(id_batch)
                for track_info in spotify_tracks:
                    if not track_info or track_info['id'] not in recco_features: continue
                    prompt = self._create_embedding_prompt(track_info, recco_features[track_info['id']])
                    embedding = self._get_embedding(prompt)
                    track_data = {"id": track_info['id'], "name": track_info['name'], "artist": track_info['artists'][0]['name'], "embedding": embedding}
                    self.db.save_track(track_data)
                    cache[track_info['id']] = Track(**track_data)
        return cache
    
    def run(self):
        logger.info("Starting playback loop...")
        while True:
            try:
                playback = self.sp.current_playback()

                if not playback or not playback.get("is_playing"):
                    if self.current_track_item:
                        duration = self.current_track_item.get('duration_ms', 0)
                        finished = duration > 0 and self.last_progress_ms > (duration * 0.98)
                        self._update_scores(self.current_track_id, skipped=not finished)
                        
                        last_id = self.current_track_id
                        self.current_track_id, self.current_track_item = None, None
                        
                        if finished and self.playlist_tracks:
                            next_id = self._get_next_track(last_id)
                            if next_id:
                                logger.info(f"▶️ Queue ended. Playing next: {self.playlist_tracks[next_id].name}")
                                self.sp.start_playback(uris=[f'spotify:track:{next_id}'])
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue

                item = playback.get("item")
                if not item or item.get('type') != 'track':
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue
                
                new_id = item["id"]
                if new_id != self.current_track_id:
                    if self.current_track_item:
                        duration = self.current_track_item.get('duration_ms', 0)
                        skipped = duration > 0 and self.last_progress_ms < (duration * 0.85)
                        self._update_scores(self.current_track_id, skipped)
                    
                    self.current_track_id, self.current_track_item = new_id, item
                    self.last_progress_ms, self.song_added_for_current_track = 0, False
                    
                    if new_id in self.playlist_tracks:
                        logger.info(f"🎶 Now Playing (Managed): '{item['name']}' by {item['artists'][0]['name']}")
                    else:
                        logger.info(f"🎶 Now Playing (Unmanaged): '{item['name']}' by {item['artists'][0]['name']}'")
                        logger.warning("Current track is not in the target playlist. Scoring will be paused.")

                self.last_progress_ms = playback.get('progress_ms', 0)

                duration = self.current_track_item.get('duration_ms', 0) if self.current_track_item else 0
                if not self.song_added_for_current_track and duration > 0 and self.last_progress_ms > (duration * 0.80):
                    self._add_new_song()
                    self.song_added_for_current_track = True

            except Exception as e:
                logger.error(f"Error in playback loop: {e}", exc_info=True)
                time.sleep(30)
            time.sleep(self.POLLING_INTERVAL_SECONDS)

    def _update_scores(self, track_id: str, skipped: bool):
        if self.current_track_item:
            logger.info(f"Evaluating: '{self.current_track_item.get('name')}' (skipped: {skipped})")
        
        if not track_id or track_id not in self.playlist_tracks:
            return

        track = self.playlist_tracks[track_id]
        track.score = track.score * self.SKIP_PENALTY if skipped else track.score + self.FINISH_BONUS
        track.skip_count += 1 if skipped else 0
        logger.info(f"{'🔻' if skipped else '🔺'} Score for '{track.name}' {'de' if skipped else 'in'}creased to {track.score:.2f}")
        
        self.db.update_track_stats(track.id, track.score, track.skip_count)
        
        if track.skip_count >= 3:
            logger.warning(f"Removing '{track.name}' from playlist due to repeated skips.")
            self.sp.remove_playlist_items(self.playlist_id, [track.id])
            del self.playlist_tracks[track_id]

    def _get_next_track(self, last_played_id: str = None) -> str:
        candidates = {tid: t for tid, t in self.playlist_tracks.items() if tid != last_played_id}
        if not candidates: return list(self.playlist_tracks.keys())[0] if self.playlist_tracks else None

        scores = np.array([t.score for t in candidates.values()])
        if scores.sum() == 0: scores = np.ones(len(candidates))

        probs = scores / scores.sum()
        return np.random.choice(list(candidates.keys()), p=probs)

    def _add_new_song(self):
        if np.random.rand() > 0.3: self._add_similar_song()
        else: self._add_generative_song()

    def _add_similar_song(self):
        if not self.current_track_id or self.current_track_id not in self.playlist_tracks: return
        current_track = self.playlist_tracks[self.current_track_id]
        
        similar_ids = self.db.find_similar_tracks(vector=current_track.embedding, n_results=5, exclude_ids=list(self.playlist_tracks.keys()))
        
        if not similar_ids:
            logger.info("[Similarity] No suitable new tracks found in database.")
            return

        candidate_id = similar_ids[0]
        candidate = self.library_tracks.get(candidate_id)
        
        if not candidate:
            newly_synced = self._sync_tracks_with_db([candidate_id])
            self.library_tracks.update(newly_synced)
            candidate = self.library_tracks.get(candidate_id)

        if candidate:
            logger.info(f"✨ [Similarity] Adding: '{candidate.name}' by {candidate.artist}")
            self.sp.add_to_playlist(self.playlist_id, [candidate.id])
            self.playlist_tracks[candidate.id] = candidate

    def _get_reccobeats_features_batch(self, ids: list[str]) -> dict:
        if not ids: return {}
        try:
            response = requests.get("https://api.reccobeats.com/v1/audio-features", params={"ids": ",".join(ids)}, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json().get('content', [])
            if len(ids) != len(data):
                logger.warning(f"ReccoBeats ID/result mismatch. Sent {len(ids)}, received {len(data)}.")
                return {}
            return {sid: f for sid, f in zip(ids, data)}
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"ReccoBeats API error: {e}")
            return {}

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            return np.array(ollama.embeddings(model=self.EMBEDDING_MODEL, prompt=text)["embedding"])
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise

    def _create_embedding_prompt(self, track: dict, features: dict) -> str:
        return f"{track['name']} by {track['artists'][0]['name']} - Tempo: {features.get('tempo')}, Danceability: {features.get('danceability')}"

    def _add_generative_song(self):
        if not self.playlist_tracks: return
        history = "\n".join([f"- '{t.name}'" for t in list(self.playlist_tracks.values())[-5:]])
        prompt = (f"Songs played:\n{history}\n\nRecommend a new song. Respond with JSON: "
                  '{"song_name": "SONG", "artist_name": "ARTIST"}')
        try:
            response = ollama.chat(model=self.GENERATIVE_MODEL, messages=[{'role': 'user', 'content': prompt}], format="json")
            content = response['message']['content']
            recommendation = json.loads(content)
            song_name, artist_name = recommendation.get('song_name'), recommendation.get('artist_name')
            if not song_name or not artist_name:
                logger.warning(f"LLM response missing keys. Raw response: {content}")
                return

            results = self.sp.search(q=f"track:{song_name} artist:{artist_name}", limit=1, type="track")
            if not results['tracks']['items']: return
            
            track = results['tracks']['items'][0]
            if track['id'] in self.playlist_tracks: return
            
            features = self._get_reccobeats_features_batch([track['id']]).get(track['id'])
            if not features: return

            embedding = self._get_embedding(self._create_embedding_prompt(track, features))
            new_track = Track(id=track['id'], name=track['name'], artist=track['artists'][0]['name'], embedding=embedding)
            
            logger.info(f"➕ [Generative] Adding: '{new_track.name}' by {new_track.artist}")
            self.sp.add_to_playlist(self.playlist_id, [new_track.id])
            self.playlist_tracks[new_track.id] = new_track
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error processing generative suggestion: {e}")
            if 'response' in locals() and response:
                logger.debug(f"Raw LLM response causing error: {response.get('message', {}).get('content', 'N/A')}")
        except Exception as e:
            logger.error(f"Unexpected error in generative suggestion: {e}", exc_info=True)