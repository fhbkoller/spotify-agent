import os
import time
import json
import ollama
import spotipy
import requests
import asyncio
import aiohttp
import numpy as np
import logging
from tqdm import tqdm
from spotipy.oauth2 import SpotifyOAuth
from dataclasses import dataclass, field
from threading import Thread, Event
from queue import Queue, Empty
from collections import deque

from persistence import PersistenceManager

# Configure logging to be less verbose for third-party libraries
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("spotipy").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- Dataclasses ---
@dataclass
class SessionContext:
    """An in-memory object to track real-time listening behavior for a session."""
    recent_history: deque = field(default_factory=lambda: deque(maxlen=5))
    total_played_count: int = 0
    cumulative_features: dict = field(default_factory=dict)

    def add_event(self, track_name: str, event_type: str, audio_features: dict | None = None):
        self.recent_history.append({"track_name": track_name, "event": event_type})
        if event_type == 'fully played' and audio_features:
            self.total_played_count += 1
            for key, value in audio_features.items():
                if isinstance(value, (int, float)):
                    self.cumulative_features[key] = self.cumulative_features.get(key, 0.0) + value

    def get_sonic_profile(self) -> dict:
        if self.total_played_count == 0:
            return {"average_sonic_profile": "N/A"}
        avg_profile = {key: round(value / self.total_played_count, 3) for key, value in self.cumulative_features.items()}
        return avg_profile

    def get_formatted_history(self) -> list[str]:
        return [f"- '{evt['track_name']}' was {evt['event']}" for evt in self.recent_history]

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
    GENERATIVE_MODEL = "llama3.1:8b-instruct-q4_k_m"
    
    # --- Agent Configuration ---
    API_BATCH_SIZE = 40
    POLLING_INTERVAL_SECONDS = 2
    SPOTIFY_QUEUE_HARD_LIMIT = 81
    RESHUFFLE_THRESHOLD = 5
    AI_REQUEST_TIMEOUT = 25 # Increased timeout for more complex models
    SKIP_PENALTY = 0.8
    FINISH_BONUS = 1.1

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
        self.session_context: SessionContext = SessionContext()

    def _setup_spotify_client(self):
        scope = (
            "user-read-playback-state user-modify-playback-state "
            "playlist-modify-public playlist-read-private user-library-read"
        )
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    def initialize(self):
        logger.info("Initializing Intelligent Shuffle Agent...")
        logger.info(f"Using embedding model: {self.EMBEDDING_MODEL}")
        logger.info(f"Using generative model: {self.GENERATIVE_MODEL}")
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
        self.playlist_tracks = self._sync_tracks_with_db(playlist_items, playlist_ids)
        logger.info(f"Loaded {len(self.playlist_tracks)} tracks for playlist.")
        
        self.session_context = SessionContext()
        self.reset_all_track_scores()
        self._reshuffle_internal_queue_only()
        
        self._worker_thread = Thread(target=self._worker_loop, name="ShufflerWorker", daemon=True)
        self._worker_thread.start()

    def _sync_tracks_with_db(self, all_track_items: list, spotify_ids: list[str]) -> dict[str, Track]:
        cache = {}
        if not spotify_ids: return cache
        found_in_db, missing_ids = self.db.get_tracks_by_ids(spotify_ids)
        
        for track_id, track_data in found_in_db.items():
            cache[track_id] = Track(
                id=track_id, name=track_data['name'], artist=track_data['artist'],
                embedding=track_data.get('embedding', np.array([]))
            )

        if missing_ids:
            logger.info(f"Found {len(cache)} tracks in DB. Fetching {len(missing_ids)} new tracks...")
            missing_track_items = [item for item in all_track_items if item.get('track') and item['track']['id'] in missing_ids]

            for batch in tqdm([missing_track_items[i:i + self.API_BATCH_SIZE] for i in range(0, len(missing_track_items), self.API_BATCH_SIZE)], desc="Analyzing new tracks"):
                batch_ids = [item['track']['id'] for item in batch]
                reccobeats_features = self._get_reccobeats_features_batch(batch_ids)
                
                if not reccobeats_features:
                    logger.warning(f"Skipping batch due to missing ReccoBeats data.")
                    continue

                for item in batch:
                    track_info = item['track']
                    track_id = track_info['id']
                    track_features = reccobeats_features.get(track_id)
                    
                    if not track_features:
                        logger.warning(f"No ReccoBeats features for track: {track_id}, {track_info['name']}, {track_info['artists'][0]['name']}. Skipping embedding.")
                        continue

                    embedding_prompt = self._create_embedding_prompt(track_info, track_features)
                    embedding = self._get_embedding(embedding_prompt)
                    
                    new_track = Track(id=track_id, name=track_info['name'], artist=track_info['artists'][0]['name'], embedding=embedding)
                    cache[track_id] = new_track
                    self.db.save_track({"id": track_id, "name": new_track.name, "artist": new_track.artist, "embedding": embedding})
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
                    if last_known_track_id and last_known_track_id in self.playlist_tracks:
                        last_track_info = self.sp.track(last_known_track_id)
                        last_duration_ms = last_track_info.get('duration_ms', 0)
                        fully_played = last_duration_ms > 0 and last_known_progress_ms >= (last_duration_ms * 0.90)
                        
                        track_name = self.playlist_tracks[last_known_track_id].name
                        logger.info(f"Track change detected. Evaluating '{track_name}' (fully_played: {fully_played})")
                        self._work_queue.put(("evaluate", {"track_id": last_known_track_id, "skipped": not fully_played}))

                    logger.info(f"🎶 Now Playing: '{playback['item']['name']}'")
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

    async def _update_scores_async(self, track_id: str, skipped: bool):
        track = self.playlist_tracks.get(track_id)
        if not track: return

        self._evaluation_counter += 1
        event = "skipped" if skipped else "fully played"
        track.last_event = event
        
        track_features = self._get_reccobeats_features_batch([track.id]).get(track.id)
        self.session_context.add_event(track.name, event, track_features)
        
        ai_response = await self._get_ai_score_multiplier_async(track, event)
        reasoning = "Default multiplier applied."
        final_multiplier = self.SKIP_PENALTY if skipped else self.FINISH_BONUS
        
        if ai_response and 'score_multiplier' in ai_response:
            ai_multiplier = ai_response['score_multiplier']
            if isinstance(ai_multiplier, (int, float)):
                if (skipped and 0.4 <= ai_multiplier <= 0.95) or (not skipped and 1.05 <= ai_multiplier <= 1.5):
                    final_multiplier = ai_multiplier
                    reasoning = ai_response.get("reasoning", "AI provided no reasoning.")
                    logger.info(f"🤖 AI provided a valid multiplier: {final_multiplier:.2f}")
                else:
                    reasoning = f"AI multiplier {ai_multiplier:.2f} was outside valid range for '{event}'."
            else:
                reasoning = "AI response was not a number."
        else:
            reasoning = "AI scoring failed or provided invalid format."
        
        original_score = track.score
        score_delta = (original_score * final_multiplier) - original_score
        track.score += score_delta
        logger.info(f"Score for '{track.name}' updated: {original_score:.2f} -> {track.score:.2f} (delta: {score_delta:+.2f}). Reason: {reasoning}")

        self._apply_similarity_score_updates(track, score_delta)

        track.skip_count += 1 if skipped else 0
        track.play_count += 1 if not skipped else 0
        
        if track.skip_count >= 3:
            logger.warning(f"Removing '{track.name}' from playlist due to {track.skip_count} skips.")
            self.sp.playlist_remove_all_occurrences_of_items(self.playlist_id, [track.id])
            del self.playlist_tracks[track.id]
            self._work_queue.put(("add_new_song", {}))

        self._reshuffle_and_update_spotify_queue()
    
    def _apply_similarity_score_updates(self, evaluated_track: Track, score_delta: float):
        if not evaluated_track.embedding.any():
            return

        logger.info(f"Applying similarity scoring based on '{evaluated_track.name}'...")
        logger.debug("--- Start Similarity Scoring Debug ---")
        
        similarities = []
        for other_track in self.playlist_tracks.values():
            if other_track.id == evaluated_track.id or not other_track.embedding.any():
                continue
            similarity = self._cosine_similarity(evaluated_track.embedding, other_track.embedding)
            if similarity > 0.5:
                similarities.append((other_track, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not similarities:
            logger.info(" -> No tracks found above the similarity threshold.")
            logger.debug("No tracks found above the similarity threshold.")
            logger.debug("--- End Similarity Scoring Debug ---")
            return

        for similar_track, similarity_score in similarities[:5]:
            adjustment = score_delta * similarity_score * 0.8
            original_sim_score = similar_track.score
            similar_track.score = max(0.01, similar_track.score + adjustment)

            logger.info(f"    -> Similar track '{similar_track.name}' adjustment: {adjustment:+.3f}, new score: {similar_track.score:.2f}")
            logger.debug(f"  - Adjusting '{similar_track.name}' based on similarity to '{evaluated_track.name}'.")
            logger.debug(f"    - Cosine Similarity: {similarity_score:.4f}")
            logger.debug(f"    - Score Adj: {adjustment:+.4f}, Score: {original_sim_score:.2f} -> {similar_track.score:.2f}")

        logger.debug("--- End Similarity Scoring Debug ---")


    def _reshuffle_and_update_spotify_queue(self):
        self._reshuffle_internal_queue_only()
        
        if self._eliminated_track_count > 0:
            logger.info(f"Queueing generative search to replace {self._eliminated_track_count} eliminated tracks.")
            for _ in range(self._eliminated_track_count):
                self._work_queue.put(("add_new_song", {}))
            self._eliminated_track_count = 0

        if self._evaluation_counter >= self.RESHUFFLE_THRESHOLD:
            logger.info("Reshuffle threshold reached. Updating Spotify's live queue.")
            self._enqueue_next_tracks()
            self._evaluation_counter = 0

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

        self.playback_queue = ([t.id for t in remaining] + [t.id for t in skipped] + [t.id for t in newly_added] + [t.id for t in played])
        
        for track in newly_added: track.is_new = False
        logger.info("Internal playback queue has been re-sorted.")

    def _enqueue_next_tracks(self):
        if not self.playback_queue:
            logger.warning("Internal queue is empty; cannot update Spotify.")
            return

        try:
            queue_info = self.sp.queue()
            current_queue_ids = {item['id'] for item in queue_info['queue']}
            
            for track_id in self.playback_queue:
                if track_id not in current_queue_ids:
                    self.sp.add_to_queue(f"spotify:track:{track_id}")
            logger.info(f"Updated Spotify queue.")
        except Exception as e:
            logger.error(f"Failed to update Spotify queue: {e}", exc_info=True)
    
    def _build_generative_prompt(self, positive_seed_tracks: list[Track], negative_seed_tracks: list[Track], session_profile: dict) -> str:
        artists_to_exclude = {t.artist for t in positive_seed_tracks}
        positive_examples = "\n".join([f"- \"{t.name}\" by {t.artist}" for t in positive_seed_tracks])
        negative_examples = "\n".join([f"- \"{t.name}\" by {t.artist}" for t in negative_seed_tracks]) or "None"
        profile_str = json.dumps(session_profile, indent=2)

        return f"""
**Role**: You are a world-class DJ. Recommend ONE perfect song to continue the current vibe.
**Analysis of Session**:
1. **Songs the User is LIKING**:
{positive_examples}
2. **Songs the User is DISLIKING**:
{negative_examples}
3. **Calculated Sonic Profile of the Session**:
{profile_str}
**Task**: Based on your analysis, recommend ONE new song.
**Strict Rules**:
1. MUST match the vibe of the positive anchors and sonic profile.
2. MUST be different from the negative anchors.
3. MUST NOT be by any of these artists: {', '.join(artists_to_exclude)}.
4. MUST be a well-known track.
**Output (JSON ONLY)**:
{{"analysis": "Brief analysis of your choice.", "song_name": "SONG_TITLE", "artist_name": "ARTIST_NAME"}}
"""

    async def _add_generative_song_async(self):
        logger.info("Attempting to add a new song with generative AI...")
        if not self.playlist_tracks: return

        positive_seeds = sorted(self.playlist_tracks.values(), key=lambda t: t.score, reverse=True)[:5]
        negative_seeds = sorted([t for t in self.playlist_tracks.values() if t.last_event == 'skipped'], key=lambda t: t.score)[:3]
        session_profile = self.session_context.get_sonic_profile()
        prompt = self._build_generative_prompt(positive_seeds, negative_seeds, session_profile)

        recommendation = await self._ollama_chat_async(prompt)
        
        if not recommendation or 'song_name' not in recommendation or 'artist_name' not in recommendation:
            logger.error(f"Failed to parse LLM response for generative song. Response: {recommendation}")
            return

        song_name, artist_name = recommendation['song_name'], recommendation['artist_name']
        logger.info(f"🤖 LLM suggested: '{song_name}' by {artist_name}. Analysis: {recommendation.get('analysis', 'N/A')}")
        
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
            logger.warning(f"Could not get ReccoBeats data for LLM suggestion. Skipping.")
            return

        embedding = self._get_embedding(self._create_embedding_prompt(new_track_info, reccobeats_data))
        new_track = Track(id=new_track_info['id'], name=new_track_info['name'], artist=new_track_info['artists'][0]['name'], embedding=embedding, is_new=True)
        
        logger.info(f"➕ [Generative] Adding: '{new_track.name}' by {new_track.artist}")
        self.sp.playlist_add_items(self.playlist_id, [new_track.id])
        self.playlist_tracks[new_track.id] = new_track
        self.db.save_track({"id": new_track.id, "name": new_track.name, "artist": new_track.artist, "embedding": embedding})

    async def _get_ai_score_multiplier_async(self, track: Track, event: str) -> dict | None:
        logger.info(f"Getting AI score multiplier for '{track.name}'...")
        history_summary = ", ".join([f"'{evt['track_name']}' was {evt['event']}" for evt in self.session_context.recent_history])
        profile_summary = ", ".join([f"{k.replace('average_', '')}:{v}" for k,v in self.session_context.get_sonic_profile().items()])

        prompt = f"""
**Analyze**:
- **Track**: "{track.name}" by {track.artist}
- **Action**: **{event}**
- **Session History**: {history_summary}
- **Session Vibe**: {profile_summary}
**Task**: Compute a score multiplier.
**Rules**:
- Skipped: 0.4-0.95.
- Fully Played: 1.05-1.5.
- Align multiplier with session history and vibe.
**Output (JSON ONLY)**:
{{"reasoning": "Brief analysis.", "score_multiplier": <float>}}
"""
        return await self._ollama_chat_async(prompt)

    async def _ollama_chat_async(self, prompt: str) -> dict | None:
        payload = {"model": self.GENERATIVE_MODEL, "messages": [{"role": "user", "content": prompt}], "format": "json", "stream": False}
        raw_content = ""
        try:
            timeout = aiohttp.ClientTimeout(total=self.AI_REQUEST_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post("http://127.0.0.1:11434/api/chat", json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    raw_content = result.get('message', {}).get('content', '')
                    return json.loads(raw_content)
        except asyncio.TimeoutError:
            logger.error(f"Ollama request timed out after {self.AI_REQUEST_TIMEOUT} seconds.")
            return None
        except json.JSONDecodeError:
            logger.error(f"Ollama returned malformed JSON! Raw content: {raw_content}")
            return None
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return None

    def _get_reccobeats_features_batch(self, spotify_track_ids: list[str]) -> dict:
        if not spotify_track_ids: return {}
        try:
            response = requests.get("https://api.reccobeats.com/v1/audio-features", params={"ids": ",".join(spotify_track_ids)}, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json()
            if 'content' in data and isinstance(data['content'], list):
                return {sid: features for sid, features in zip(spotify_track_ids, data['content'])}
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get audio features from ReccoBeats: {e}")
            return {}

    def _create_embedding_prompt(self, track_info: dict, reccobeats_data: dict) -> str:
        features_str = ", ".join([f"{key}: {value}" for key, value in reccobeats_data.items() if key not in ['title', 'spotify_id']])
        return f"artist: {track_info['artists'][0]['name']}, features: {features_str}"

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            return np.array(ollama.embeddings(model=self.EMBEDDING_MODEL, prompt=text)["embedding"])
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            return np.zeros(384)
    
    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)