import os
import time
import json
import ollama
import spotipy
import asyncio
import aiohttp
import numpy as np
import logging
from tqdm import tqdm
from spotipy.oauth2 import SpotifyOAuth
from dataclasses import dataclass, field
from threading import Thread, Event
from queue import Queue, Empty

from persistence import PersistenceManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Track:
    id: str
    name: str
    artist: str
    embedding: np.ndarray = field(repr=False)
    score: float = 1.0
    skip_count: int = 0
    play_count: int = 0
    last_event: str = 'fresh'
    is_new: bool = False

class IntelligentShuffler:
    EMBEDDING_MODEL = "mxbai-embed-large"
    GENERATIVE_MODEL = "qwen:7b"
    API_BATCH_SIZE = 40
    POLLING_INTERVAL_SECONDS = 2
    SPOTIFY_QUEUE_HARD_LIMIT = 81
    RESHUFFLE_THRESHOLD = 5
    AI_REQUEST_TIMEOUT = 15

    def __init__(self, playlist_id: str):
        self._setup_spotify_client()
        self.playlist_id = playlist_id
        self.user_id = self.sp.me()["id"]
        self.db = PersistenceManager()
        self.playlist_tracks: dict[str, Track] = {}
        self.playback_queue: list[str] = []
        self._work_queue: Queue = Queue()
        self._stop_event: Event = Event()
        self._worker_thread: Thread | None = None
        self._evaluation_counter = 0
        self._eliminated_track_count = 0

    def _setup_spotify_client(self):
        scope = (
            "user-read-playback-state user-modify-playback-state "
            "playlist-modify-public playlist-read-private user-library-read"
        )
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    def initialize(self):
        logger.info("Initializing Intelligent Shuffle Agent...")
        logger.info("Syncing tracks from target playlist...")
        playlist_items = []
        offset = 0
        while True:
            response = self.sp.playlist_items(self.playlist_id, offset=offset, limit=100)
            if not response or not response.get("items"): break
            items = response["items"]
            playlist_items.extend(items)
            if len(items) < 100: break
            offset += 100
        
        playlist_ids = [item['track']['id'] for item in playlist_items if item.get('track') and item.get('track').get('id')]
        self.playlist_tracks = self._sync_tracks_with_db(playlist_ids)
        logger.info(f"Loaded {len(self.playlist_tracks)} tracks for playlist.")
        
        self.reset_all_track_scores()
        self._reshuffle_internal_queue_only()
        
        self._worker_thread = Thread(target=self._worker_loop, name="ShufflerWorker", daemon=True)
        self._worker_thread.start()

    def reset_all_track_scores(self):
        logger.info("Resetting all in-memory track scores and states for the new session.")
        for track in self.playlist_tracks.values():
            track.score = 1.0
            track.skip_count = 0
            track.play_count = 0
            track.last_event = 'fresh'
            track.is_new = False

    def _reshuffle_internal_queue_only(self):
        if not self.playlist_tracks:
            self.playback_queue = []
            return
        
        active_tracks = {tid: t for tid, t in self.playlist_tracks.items() if t.score > 0}
        
        eliminated_count = len(self.playlist_tracks) - len(active_tracks)
        if eliminated_count > 0:
            self._eliminated_track_count += eliminated_count
            logger.info(f"{eliminated_count} tracks were eliminated due to low score. Total to replace: {self._eliminated_track_count}")
            self.playlist_tracks = active_tracks

        remaining = sorted([t for t in active_tracks.values() if t.last_event == 'fresh'], key=lambda t: t.score, reverse=True)
        skipped = sorted([t for t in active_tracks.values() if t.last_event == 'skipped'], key=lambda t: t.score, reverse=True)
        played = sorted([t for t in active_tracks.values() if t.last_event == 'played'], key=lambda t: t.score, reverse=True)
        newly_added = [t for t in active_tracks.values() if t.is_new]

        self.playback_queue = (
            [t.id for t in remaining] + [t.id for t in skipped] +
            [t.id for t in newly_added] + [t.id for t in played]
        )
        
        for track in newly_added: track.is_new = False
        logger.info("Internal playback queue has been re-sorted.")

    def _reshuffle_and_update_spotify_queue(self):
        self._reshuffle_internal_queue_only()
        
        if self._eliminated_track_count > 0:
            logger.info(f"Queueing generative search to replace {self._eliminated_track_count} eliminated tracks.")
            for _ in range(self._eliminated_track_count):
                self._work_queue.put(("add_new_song", {}))
            self._eliminated_track_count = 0

        if self._evaluation_counter >= self.RESHUFFLE_THRESHOLD:
            logger.info("Reshuffle threshold reached. Replacing Spotify's live queue.")
            self._enqueue_next_tracks()
            self._evaluation_counter = 0

    def _sync_tracks_with_db(self, spotify_ids: list[str]) -> dict[str, Track]:
        cache = {}
        if not spotify_ids: return cache
        found_tracks_data, missing_ids = self.db.get_tracks_by_ids(spotify_ids)
        
        for track_id, track_data in found_tracks_data.items():
            embedding = track_data.get('embedding')
            cache[track_id] = Track(id=track_id, name=track_data['name'], artist=track_data['artist'], embedding=embedding if embedding is not None else np.zeros(384))

        ids_to_fetch = list(set(missing_ids))
        if ids_to_fetch:
            for id_batch in tqdm([ids_to_fetch[i:i + self.API_BATCH_SIZE] for i in range(0, len(ids_to_fetch), self.API_BATCH_SIZE)], desc="Syncing new tracks"):
                for track_info in self.sp.tracks(id_batch)['tracks']:
                    if not track_info: continue
                    tid, tname, tartist = track_info['id'], track_info['name'], track_info['artists'][0]['name']
                    embedding = self._get_embedding(f"{tname} by {tartist}")
                    track_data = {"id": tid, "name": tname, "artist": tartist, "embedding": embedding}
                    self.db.save_track(track_data)
                    cache[tid] = Track(id=tid, name=tname, artist=tartist, embedding=embedding)
        return cache

    def run(self):
        logger.info("Starting playback loop...")
        last_known_track_id, last_known_progress_ms = None, 0

        try:
            initial_playback = self.sp.current_playback()
            if initial_playback and initial_playback.get('item'):
                last_known_track_id = initial_playback['item']['id']
                logger.info(f"Initial track detected: '{initial_playback['item']['name']}'")
        except Exception as e:
            logger.error(f"Could not get initial playback state: {e}")

        while not self._stop_event.is_set():
            try:
                playback = self.sp.current_playback()
                if not playback or not playback.get("is_playing") or not playback.get("item"):
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue

                current_track_id = playback['item']['id']
                
                if current_track_id != last_known_track_id:
                    track_for_logging = self.playlist_tracks.get(last_known_track_id, 
                                                                 Track(id=last_known_track_id, name=f"ID: {last_known_track_id}", artist="Unknown", embedding=np.array([])))
                    track_name = track_for_logging.name
                    logger.info(f"Track change detected: Now playing '{playback['item']['name']}'")
                    
                    if last_known_track_id in self.playlist_tracks:
                        last_track_info = self.sp.track(last_known_track_id)
                        last_duration_ms = last_track_info.get('duration_ms', 0)
                        fully_played = last_duration_ms > 0 and last_known_progress_ms >= (last_duration_ms * 0.95)
                        
                        logger.info(f"Queueing evaluation for previous track '{track_name}' (fully_played: {fully_played})")
                        self._work_queue.put(("evaluate", {"track_id": last_known_track_id, "skipped": not fully_played}))

                    last_known_track_id = current_track_id
                
                last_known_progress_ms = playback.get('progress_ms', 0)

            except Exception as e:
                logger.error(f"Error in playback loop: {e}", exc_info=True)
                time.sleep(10)
            time.sleep(self.POLLING_INTERVAL_SECONDS)

    def _worker_loop(self):
        logger.info("Background worker started.")
        asyncio.run(self._async_worker())

    async def _async_worker(self):
        while not self._stop_event.is_set():
            try:
                task, payload = self._work_queue.get_nowait()
                try:
                    if task == "evaluate":
                        await self._update_scores_async(payload["track_id"], skipped=payload["skipped"])
                    elif task == "add_new_song":
                        await self._add_generative_song_async()
                except Exception as e:
                    logger.error(f"Error processing worker task '{task}': {e}", exc_info=True)
                finally:
                    self._work_queue.task_done()
            except Empty:
                await asyncio.sleep(0.1)

    async def _ollama_chat_async(self, prompt: str) -> dict | None:
        payload = {"model": self.GENERATIVE_MODEL, "messages": [{"role": "user", "content": prompt}], "format": "json", "stream": False}
        try:
            timeout = aiohttp.ClientTimeout(total=self.AI_REQUEST_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post("http://127.0.0.1:11434/api/chat", json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    raw_content = result.get('message', {}).get('content', '')
                    clean_content = raw_content.strip().replace("'", '"').replace('`', '')
                    if "}" not in clean_content and "{" in clean_content: clean_content += "}"
                    return json.loads(clean_content)
        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after {self.AI_REQUEST_TIMEOUT} seconds.")
            return None
        except json.JSONDecodeError:
            logger.error(f"Ollama returned malformed JSON! Raw content: {raw_content}")
            return None
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return None

    async def _get_ai_score_multiplier_async(self, track: Track, event: str) -> dict | None:
        logger.info(f"Attempting to get AI score multiplier for '{track.name}'...")
        prompt = f"""
**Instruction**: Compute a score multiplier for a song based on user action.
**Data**:
- Action: **{event}**
- History: Played {track.play_count} times.
**Rules**:
1. If "Action" is "skipped", multiplier MUST be between 0.7 and 0.9.
2. If "Action" is "fully played", multiplier MUST be between 1.05 and 1.2.
**Output (JSON ONLY)**:
{{"score_multiplier": <float>}}
"""
        return await self._ollama_chat_async(prompt)

    def _normalize_distance_to_similarity(self, distance: float, min_dist: float, max_dist: float) -> float:
        """Converts a raw distance to a 0-1 similarity score."""
        if max_dist == min_dist:
            return 1.0  # All distances are the same, so they are all equally similar
        # Invert the score so that smaller distances are closer to 1.0
        return 1.0 - ((distance - min_dist) / (max_dist - min_dist))

    async def _update_scores_async(self, track_id: str, skipped: bool):
        track = self.playlist_tracks.get(track_id)
        if not track: return

        self._evaluation_counter += 1
        event = "skipped" if skipped else "fully played"
        track.last_event = event
        logger.info(f"Evaluating: '{track.name}' ({event}). Eval count: {self._evaluation_counter}")

        final_multiplier = 0.8 if skipped else 1.1
        ai_response = await self._get_ai_score_multiplier_async(track, event)
        
        if ai_response and 'score_multiplier' in ai_response:
            ai_multiplier = ai_response['score_multiplier']
            if (skipped and ai_multiplier < 1.0) or (not skipped and ai_multiplier > 1.0):
                final_multiplier = ai_multiplier
                logger.info(f"🤖 AI provided a valid multiplier: {final_multiplier:.2f}")
            else:
                logger.warning(f"AI returned an invalid multiplier ({ai_multiplier:.2f} for a '{event}' event). Using fallback.")
        else:
            logger.warning("AI scoring failed or timed out. Applying default fallback scoring.")
        
        original_score = track.score
        new_score = original_score * final_multiplier
        score_delta = new_score - original_score
        track.score = new_score
        
        track.skip_count += 1 if skipped else 0
        track.play_count += 0 if skipped else 1
        logger.info(f"Adjustment for '{track.name}': score changed by {score_delta:+.3f}, new score={track.score:.2f}")

        if track.embedding.any():
            logger.debug("--- Start Similarity Scoring Debug ---")
            logger.debug(f"Primary track '{track.name}': original_score={original_score:.3f}, new_score={new_score:.3f}, score_delta={score_delta:+.3f}")
            
            similar_ids_and_distances = self.db.find_similar_tracks(vector=track.embedding, n_results=3, include_distances=True, exclude_ids=[track_id])
            
            dist_list = [(sid, dist) for sid, dist in similar_ids_and_distances if sid in self.playlist_tracks]
            if dist_list:
                min_dist = min(d for _, d in dist_list)
                max_dist = max(d for _, d in dist_list)
                
                for sid, dist in dist_list:
                    sim_track = self.playlist_tracks[sid]
                    similarity = self._normalize_distance_to_similarity(dist, min_dist, max_dist)
                    adjustment = score_delta * similarity * 0.5

                    logger.debug(f"  -> Similar track '{sim_track.name}':")
                    logger.debug(f"     Raw Distance = {dist:.3f} (min={min_dist:.3f}, max={max_dist:.3f})")
                    logger.debug(f"     Normalized Similarity = {similarity:.3f}")
                    logger.debug(f"     Adjustment = (delta){score_delta:+.3f} * (similarity){similarity:.3f} * 0.5 = {adjustment:+.3f}")
                    
                    sim_track.score += adjustment
                    logger.info(f"  -> Similar track '{sim_track.name}' adjustment: {adjustment:+.3f}, new score: {sim_track.score:.2f}")
            logger.debug("--- End Similarity Scoring Debug ---")
        
        self._reshuffle_and_update_spotify_queue()

    async def _add_generative_song_async(self, max_retries: int = 3):
        if not self.playlist_tracks: return
        
        for attempt in range(max_retries):
            logger.info(f"Generative search attempt {attempt + 1}/{max_retries}...")
            seed_tracks = sorted(self.playlist_tracks.values(), key=lambda t: t.score, reverse=True)[:5]
            prompt = self._build_generative_prompt(seed_tracks)
            if not prompt: return

            recommendation = await self._ollama_chat_async(prompt)
            
            song_name, artist_name = None, None
            if recommendation:
                song_name = recommendation.get('song_name') or recommendation.get('song_ name')
                artist_name = recommendation.get('artist_name') or recommendation.get('artist_ name')

            if not song_name or not artist_name:
                logger.warning(f"Generative LLM returned invalid data on attempt {attempt + 1}.")
                continue

            logger.info(f"AI suggested: '{song_name}' by '{artist_name}'. Searching on Spotify...")
            results = self.sp.search(q=f"track:{song_name} artist:{artist_name}", limit=1, type="track")
            
            if not results['tracks']['items']:
                logger.warning(f"No Spotify results for the suggested track. Retrying...")
                continue

            track, track_id = results['tracks']['items'][0], results['tracks']['items'][0]['id']
            if track_id in self.playlist_tracks:
                logger.info(f"Skipping '{track['name']}' as it's already in the playlist. Retrying...")
                continue

            embedding = self._get_embedding(f"{track['name']} by {track['artists'][0]['name']}")
            new_track = Track(id=track_id, name=track['name'], artist=track['artists'][0]['name'], embedding=embedding, is_new=True)
            self.db.save_track({"id": track_id, "name": new_track.name, "artist": new_track.artist, "embedding": embedding})
            self.playlist_tracks[track_id] = new_track
            
            self.sp.playlist_add_items(self.playlist_id, [track_id])
            self.sp.add_to_queue(f"spotify:track:{track_id}")
            logger.info(f"➕ [Generative] Added '{new_track.name}' to playlist and current queue.")
            return

        logger.error(f"Generative search failed after {max_retries} attempts.")

    def _build_generative_prompt(self, seed_tracks: list[Track]) -> str:
        if not seed_tracks: return ""
        artists_to_exclude = {t.artist for t in seed_tracks}
        track_analysis_str = "\n".join([f"- \"{t.name}\" by {t.artist}" for t in seed_tracks])
        return f"""
**Instruction**: Recommend ONE new song.
**Seed Tracks**:
{track_analysis_str}
**Rules**:
1. The song MUST be sonically similar to the seed tracks.
2. The song MUST NOT be by any of these artists: {', '.join(artists_to_exclude)}.
3. The song MUST be a well-known track likely to be found on Spotify.
**Output (JSON ONLY)**:
{{"song_name": "SONG_TITLE", "artist_name": "ARTIST_NAME"}}
"""

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            return np.array(ollama.embeddings(model=self.EMBEDDING_MODEL, prompt=text)["embedding"])
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            return np.zeros(384)

    def _enqueue_next_tracks(self):
        if not self.playback_queue:
            logger.warning("Internal queue is empty; cannot update Spotify.")
            return

        try:
            devices = self.sp.devices()
            active_device = next((d for d in devices['devices'] if d['is_active']), None)
            if not active_device:
                logger.warning("No active Spotify device found. Cannot replace queue.")
                return
            device_id = active_device['id']

            current_playback = self.sp.current_playback()
            if not current_playback or not current_playback.get('item'):
                logger.warning("Nothing is currently playing. Cannot replace queue without interrupting.")
                return

            currently_playing_id = current_playback['item']['id']
            
            final_queue_ids = [currently_playing_id] + [tid for tid in self.playback_queue if tid != currently_playing_id]
            queue_uris = [f"spotify:track:{tid}" for tid in final_queue_ids[:self.SPOTIFY_QUEUE_HARD_LIMIT]]

            self.sp.start_playback(device_id=device_id, uris=queue_uris)
            
            if current_playback.get('progress_ms'):
                self.sp.seek_track(current_playback['progress_ms'])

            logger.info(f"Successfully replaced Spotify queue with {len(queue_uris)} tracks.")
        
        except Exception as e:
            logger.error(f"Failed to replace Spotify queue: {e}", exc_info=True)