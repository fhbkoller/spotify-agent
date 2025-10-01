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
from threading import Thread, Event, Lock
from queue import Queue, Empty

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
    POLLING_INTERVAL_SECONDS = 2
    SPOTIFY_QUEUE_HARD_LIMIT = 81
    SIMILAR_BONUS = 0.02
    SIMILAR_PENALTY = 0.01
    JUMP_PENALTY = 0.2

    def __init__(self, playlist_id: str):
        self._setup_spotify_client()
        self.playlist_id = playlist_id
        self.user_id = self.sp.me()["id"]
        self.db = PersistenceManager()
        self.playlist_tracks: dict[str, Track] = {}
        self.playback_queue: list[str] = []
        self.current_track_id = None
        self.current_track_item = None
        self.song_added_for_current_track = False
        # Async/worker fields
        self._work_queue: Queue = Queue()
        self._stop_event: Event = Event()
        self._worker_thread: Thread | None = None
        self._state_lock: Lock = Lock()
        # Debounce
        self._stable_track_counter: int = 0
        self._events_since_last_queue_refresh: int = 0
        self._QUEUE_REFRESH_EVENT_THRESHOLD: int = 1
        # Snapshot of Spotify's next/previous for adjacency
        self._last_spotify_next_id = None
        self._last_spotify_prev_id = None
        self._last_progress_ms_current = 0

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
        limit = 100
        offset = 0
        while True:
            results = self.sp.playlist_items(self.playlist_id, limit=limit, offset=offset)
            items = results.get("items", [])
            playlist_items.extend(items)
            if len(items) < limit:
                break
            offset += limit
        logger.info(f"Fetched a total of {len(playlist_items)} items from Spotify playlist.")
        playlist_ids = [item['track']['id'] for item in playlist_items if item.get('track') and item['track'].get('id')]
        self.playlist_tracks = self._sync_tracks_with_db(playlist_ids)
        logger.info(f"Loaded {len(self.playlist_tracks)} tracks for playlist.")
        self._reshuffle_queue()
        # Start background worker
        try:
            self._worker_thread = Thread(target=self._worker_loop, name="ShufflerWorker", daemon=True)
            self._worker_thread.start()
        except Exception as e:
            logger.error(f"Failed to start background worker: {e}")

    def _reshuffle_queue(self):
        # Shuffle/reshuffle the queue based on scores
        if not self.playlist_tracks:
            self.playback_queue = []
            return
        scores = np.array([t.score for t in self.playlist_tracks.values()])
        if scores.sum() == 0:
            scores = np.ones(len(self.playlist_tracks))
        probs = scores / scores.sum()
        self.playback_queue = list(np.random.choice(list(self.playlist_tracks.keys()), size=len(self.playlist_tracks), replace=False, p=probs))
        logger.info(f"Playback queue reshuffled: {[self.playlist_tracks[tid].name for tid in self.playback_queue]}")

    def _sync_tracks_with_db(self, spotify_ids: list[str]) -> dict[str, Track]:
        cache = {}
        if not spotify_ids: return cache
        found_tracks_data, missing_ids = self.db.get_tracks_by_ids(spotify_ids)
        # Separate: rows present (with or without embedding) vs truly absent rows
        ids_missing_embeddings = [tid for tid, t in found_tracks_data.items() if t.get('embedding') is None]
        truly_missing_ids = list(set(missing_ids))
        # For present rows, construct Track objects; if embedding is missing, defer embedding usage
        for track_id, track_data in found_tracks_data.items():
            embedding = track_data.get('embedding')
            if embedding is not None:
                cache[track_id] = Track(**track_data)
            else:
                # Create placeholder Track without embedding for now
                cache[track_id] = Track(id=track_data['id'], name=track_data['name'], artist=track_data['artist'], embedding=np.zeros(384))
        if found_tracks_data:
            logger.info(f"Loaded {len(found_tracks_data)} tracks from local database.")
        # Fetch remote data for tracks that are truly absent OR missing embeddings
        ids_to_fetch = list(set(truly_missing_ids + ids_missing_embeddings))
        if ids_to_fetch:
            logger.info(f"Fetching data for {len(ids_to_fetch)} tracks missing locally (new or no embedding)...")
            batches = [ids_to_fetch[i:i + self.API_BATCH_SIZE] for i in range(0, len(ids_to_fetch), self.API_BATCH_SIZE)]
            for id_batch in tqdm(batches, desc="Syncing new tracks"):
                spotify_tracks = self.sp.tracks(id_batch)['tracks']
                recco_features = self._get_reccobeats_features_batch(id_batch)
                for track_info in spotify_tracks:
                    if not track_info: continue
                    tid = track_info['id']
                    track_data = {"id": tid, "name": track_info['name'], "artist": track_info['artists'][0]['name']}
                    # Ensure row exists regardless of feature availability
                    self.db.save_track(track_data)
                    # Build prompt using ReccoBeats or Spotify features, fallback to minimal text
                    feats = recco_features.get(tid) or {}
                    try:
                        prompt = self._create_embedding_prompt(track_info, feats) if feats else f"{track_info['name']} by {track_info['artists'][0]['name']}"
                        embedding = self._get_embedding(prompt)
                        self.db.save_embedding(tid, embedding)
                        cache[tid] = Track(id=tid, name=track_info['name'], artist=track_info['artists'][0]['name'], embedding=embedding)
                    except Exception:
                        # If embedding fails, keep placeholder; will recover on next run
                        if tid not in cache:
                            cache[tid] = Track(id=tid, name=track_info['name'], artist=track_info['artists'][0]['name'], embedding=np.zeros(384))
        return cache

    def reset_queue_weights(self):
        """Reset the weights (scores) of all songs in the current queue to the initial value and update the DB."""
        for track_id in self.playback_queue:
            if track_id in self.playlist_tracks:
                track = self.playlist_tracks[track_id]
                track.score = 1.0
                track.skip_count = 0
                self.db.update_track_stats(track.id, track.score, track.skip_count)
        logger.info("All queue track weights reset to 1.0.")

    def run(self):
        logger.info("Starting playback loop...")
        # Do not auto-start; only monitor
        playback = self.sp.current_playback()
        last_track_id = self.current_track_id
        last_progress_ms = 0
        last_duration_ms = 0
        previous_next_spotify_id = None  # next track snapshot from previous poll
        while True:
            try:
                playback = self.sp.current_playback()
                if not playback or not playback.get("is_playing"):
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue
                item = playback.get("item")
                if not item or item.get('type') != 'track':
                    time.sleep(self.POLLING_INTERVAL_SECONDS)
                    continue
                new_id = item["id"]
                progress_ms = playback.get('progress_ms', 0)
                duration = item.get('duration_ms', 0)
                # Pull Spotify queue to capture the NEXT track for this poll
                current_next_spotify_id = None
                try:
                    get_queue = getattr(self.sp, 'current_user_queue', None) or getattr(self.sp, 'queue', None)
                    if callable(get_queue):
                        q = get_queue()
                        queue_items = q.get('queue', []) if isinstance(q, dict) else []
                        if queue_items:
                            current_next_spotify_id = queue_items[0].get('id')
                except Exception:
                    pass
                # Pull Spotify adjacency (previous and next) every poll
                try:
                    get_queue = getattr(self.sp, 'current_user_queue', None) or getattr(self.sp, 'queue', None)
                    if callable(get_queue):
                        q = get_queue()
                        queue_items = q.get('queue', []) if isinstance(q, dict) else []
                        self._last_spotify_next_id = (queue_items[0].get('id') if queue_items else None)
                        # Previous is not provided directly; best-effort: currently_playing from previous poll
                        # We'll keep previous track id via last_track_id already
                except Exception:
                    pass
                # Detect skip-back: same track, but progress resets to near zero
                skip_back = (
                    new_id == last_track_id and progress_ms < 5000 and last_progress_ms > 10000
                )
                # Detect skip to previous track
                skip_to_previous = (
                    new_id != last_track_id and self.playback_queue and self.playback_queue and self.playback_queue[-1] == new_id
                )
                # Determine event classification based on the PREVIOUS track's progress/duration
                prev_duration_ms = last_duration_ms
                fully_played_prev = new_id != last_track_id and (prev_duration_ms > 0 and last_progress_ms >= (prev_duration_ms * 0.9))
                # Adjacent if the new track equals the next track snapshot captured on the PREVIOUS poll
                is_adjacent_next_spotify = bool(previous_next_spotify_id and new_id == previous_next_spotify_id)
                # Jump occurs when user selects a non-adjacent track (not next/previous) without full play
                jump = (new_id != last_track_id and not skip_back and not skip_to_previous and not fully_played_prev and not is_adjacent_next_spotify)
                if new_id != last_track_id:
                    if not skip_to_previous and not skip_back:
                        # Defer evaluation to worker; decide skipped vs jumped here
                        if self.current_track_item:
                            played_duration = last_progress_ms
                            skipped = (not fully_played_prev) and (is_adjacent_next_spotify or skip_to_previous)
                            try:
                                self._work_queue.put(("evaluate", {"track_id": last_track_id, "skipped": bool(skipped), "jumped": bool(jump)}))
                            except Exception:
                                pass
                    # Defer pruning and LLM fill to worker
                    try:
                        self._work_queue.put(("prune_and_fill", {}))
                    except Exception:
                        pass
                    # Pop the next track from the queue if it matches new_id
                    if self.playback_queue and self.playback_queue[0] == new_id:
                        self.playback_queue.pop(0)
                    self.current_track_id, self.current_track_item = new_id, item
                    last_progress_ms, self.song_added_for_current_track = 0, False
                    # Debounced reorder/enqueue via worker only if playing target playlist
                    try:
                        context = playback.get('context') or {}
                        ctx_uri = context.get('uri')
                        is_target_playlist = isinstance(ctx_uri, str) and self.playlist_id in ctx_uri
                    except Exception:
                        is_target_playlist = False
                    if is_target_playlist:
                        self._events_since_last_queue_refresh += 1
                        try:
                            self._work_queue.put(("maybe_reorder", {"current_id": new_id, "reason": "track_change"}))
                        except Exception:
                            pass
                    if new_id in self.playlist_tracks:
                        logger.info(f"🎶 Now Playing (Managed): '{item['name']}' by {item['artists'][0]['name']}")
                    else:
                        logger.info(f"🎶 Now Playing (Unmanaged): '{item['name']}' by {item['artists'][0]['name']}'")
                        logger.warning("Current track is not in the target playlist. Scoring will be paused.")
                last_progress_ms = progress_ms
                last_duration_ms = duration
                last_track_id = new_id
                # Update previous_next_spotify_id for the next loop iteration
                previous_next_spotify_id = current_next_spotify_id
                if not self.song_added_for_current_track and duration > 0 and progress_ms > (duration * 0.80):
                    try:
                        self._work_queue.put(("add_new_song", {}))
                        self.song_added_for_current_track = True
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error in playback loop: {e}", exc_info=True)
                time.sleep(30)
            time.sleep(self.POLLING_INTERVAL_SECONDS)

    def _worker_loop(self):
        logger.info("Background worker started.")
        while not getattr(self, "_stop_event", Event()).is_set():
            try:
                task, payload = self._work_queue.get(timeout=1)
            except Exception:
                continue
            try:
                if task == "enqueue":
                    self._enqueue_next_tracks()
                elif task == "evaluate":
                    tid = payload.get("track_id")
                    skipped_flag = payload.get("skipped")
                    jumped_flag = payload.get("jumped")
                    if tid is not None:
                        self._update_scores(tid, skipped=bool(skipped_flag), jumped=bool(jumped_flag))
                elif task == "prune_and_fill":
                    to_remove = [tid for tid in self.playback_queue if tid in self.playlist_tracks and self.playlist_tracks[tid].score <= 0]
                    for tid in to_remove:
                        try:
                            logger.info(f"Removing '{self.playlist_tracks[tid].name}' from queue due to score <= 0.")
                        except KeyError:
                            pass
                        self.playback_queue = [x for x in self.playback_queue if x != tid]
                        if tid in self.playlist_tracks:
                            del self.playlist_tracks[tid]
                        self._add_llm_suggestion_based_on_top_scores()
                elif task == "maybe_reorder":
                    current_id = payload.get("current_id")
                    if self._events_since_last_queue_refresh >= getattr(self, "_QUEUE_REFRESH_EVENT_THRESHOLD", 5):
                        self._reorder_queue_after_current(current_id)
                        self._enqueue_next_tracks()
                        self._events_since_last_queue_refresh = 0
                elif task == "add_new_song":
                    self._add_new_song()
            except Exception as e:
                logger.error(f"Worker task {task} failed: {e}", exc_info=True)
            finally:
                try:
                    self._work_queue.task_done()
                except Exception:
                    pass

    def _update_scores(self, track_id: str, skipped: bool, jumped: bool = False):
        # Log evaluated track's actual name, not the current playing item
        eval_name = self.playlist_tracks.get(track_id).name if track_id in self.playlist_tracks else track_id
        logger.info(f"Evaluating: '{eval_name}' (skipped: {skipped}, jumped: {jumped})")
        
        if not track_id or track_id not in self.playlist_tracks:
            return

        track = self.playlist_tracks[track_id]
        if jumped:
            track.score = track.score * self.JUMP_PENALTY
            setattr(track, 'was_jumped', True)
            logger.info(f"⏩ Score for '{track.name}' jumped to {track.score:.2f}")
        elif skipped:
            track.score = track.score * self.SKIP_PENALTY
            track.skip_count += 1
            logger.info(f"🔻 Score for '{track.name}' decreased to {track.score:.2f}")
            # Decrement similar songs
            similar_ids = self.db.find_similar_tracks(vector=track.embedding, n_results=5, exclude_ids=[track_id])
            for sid in similar_ids:
                if sid in self.playlist_tracks:
                    similar_track = self.playlist_tracks[sid]
                    similar_track.score -= self.SIMILAR_PENALTY
                    self.db.update_track_stats(similar_track.id, similar_track.score, similar_track.skip_count)
                    logger.info(f"🔸 Similar track '{similar_track.name}' received penalty, new score: {similar_track.score:.2f}")
        else:
            track.score = track.score + self.FINISH_BONUS
            setattr(track, 'played_full', True)
            logger.info(f"🔺 Score for '{track.name}' increased to {track.score:.2f}")
            # Increment similar songs
            similar_ids = self.db.find_similar_tracks(vector=track.embedding, n_results=5, exclude_ids=[track_id])
            for sid in similar_ids:
                if sid in self.playlist_tracks:
                    similar_track = self.playlist_tracks[sid]
                    similar_track.score += self.SIMILAR_BONUS
                    self.db.update_track_stats(similar_track.id, similar_track.score, similar_track.skip_count)
                    logger.info(f"🔹 Similar track '{similar_track.name}' received bonus, new score: {similar_track.score:.2f}")
        self.db.update_track_stats(track.id, track.score, track.skip_count)
        if track.skip_count >= 3:
            logger.warning(f"Removing '{track.name}' from queue due to repeated skips.")
            # Remove from queue if present, but do not touch the playlist
            self.playback_queue = [tid for tid in self.playback_queue if tid != track_id]
            del self.playlist_tracks[track_id]

    def _get_next_track(self, last_played_id: str = None) -> str:
        # Pop the next track from the queue, reshuffle if empty
        if not self.playback_queue:
            self._reshuffle_queue()
        if not self.playback_queue:
            return None
        next_id = self.playback_queue.pop(0)
        return next_id

    def _add_new_song(self):
        if np.random.rand() > 0.3: self._add_similar_song()
        else: self._add_generative_song()

    def _add_similar_song(self):
        if not self.current_track_id or self.current_track_id not in self.playlist_tracks: return
        current_track = self.playlist_tracks[self.current_track_id]
        
        # If current track has placeholder embedding (zeros), do not attempt similarity
        if current_track.embedding is None or (isinstance(current_track.embedding, np.ndarray) and not current_track.embedding.any()):
            logger.info("[Similarity] Current track missing embedding; skipping similarity search.")
            return
        similar_ids = self.db.find_similar_tracks(vector=current_track.embedding, n_results=10, exclude_ids=list(self.playlist_tracks.keys()))
        
        if not similar_ids:
            logger.info("[Similarity] No suitable new tracks found in database.")
            return
        
        # Prefer the highest scoring among the similar candidates we know
        candidate_id = None
        for sid in similar_ids:
            if sid in self.playlist_tracks:
                candidate_id = sid
                break
        if candidate_id is None:
            candidate_id = similar_ids[0]
        newly_synced = self._sync_tracks_with_db([candidate_id])
        candidate = newly_synced.get(candidate_id)
        
        if candidate:
            logger.info(f"✨ [Similarity] Adding to queue: '{candidate.name}' by {candidate.artist}")
            self.playback_queue.append(candidate.id)
            self.playlist_tracks[candidate.id] = candidate  # Optionally track for scoring

    def _get_reccobeats_features_batch(self, ids: list[str]) -> dict:
        if not ids: return {}
        try:
            response = requests.get("https://api.reccobeats.com/v1/audio-features", params={"ids": ",".join(ids)}, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json().get('content', [])
            if len(ids) != len(data):
                logger.warning(f"ReccoBeats ID/result mismatch. Sent {len(ids)}, received {len(data)}.")
                # Proceed with partial mapping instead of dropping the whole batch
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
        # Weighted features prompt; exclude title, de-emphasize artist
        artist = track['artists'][0]['name']
        tempo = features.get('tempo')
        dance = features.get('danceability')
        energy = features.get('energy') if isinstance(features, dict) else None
        valence = features.get('valence') if isinstance(features, dict) else None
        parts = []
        if tempo is not None:
            parts.append(f"tempo:{tempo}")
        if dance is not None:
            parts.append(f"danceability:{dance}")
        if energy is not None:
            parts.append(f"energy:{energy}")
        if valence is not None:
            parts.append(f"valence:{valence}")
        # Add low-weight artist context
        parts.append(f"artist:{artist}")
        return " ".join(parts)

    def _build_generative_prompt(self, seed_tracks: list[Track]) -> str:
        """Builds a detailed, context-rich prompt for the generative model."""
        if not seed_tracks:
            return ""

        # Fetch audio features for the seed tracks to enrich the prompt
        seed_ids = [t.id for t in seed_tracks]
        features_map = self._get_reccobeats_features_batch(seed_ids)
        
        # Build the detailed analysis section of the prompt
        track_analysis_str = ""
        artists_to_exclude = set()
        for track in seed_tracks:
            artists_to_exclude.add(track.artist)
            features = features_map.get(track.id)
            track_analysis_str += f"* **Track:** \"{track.name}\" by {track.artist}\n"
            if features:
                track_analysis_str += (
                    f"    * **Key Features:** "
                    f"danceability={features.get('danceability', 'N/A')}, "
                    f"energy={features.get('energy', 'N/A')}, "
                    f"valence={features.get('valence', 'N/A')}\n"
                )

        prompt = f"""You are an expert music curator with deep knowledge of genres, moods, and sonic textures.
Your task is to recommend ONE new song that is sonically and thematically similar to the following tracks.
Do not recommend another song by any of the following artists: {', '.join(artists_to_exclude)}.

**Analysis of Provided Tracks:**
{track_analysis_str}
**Your Recommendation:**
Based on this analysis, recommend one new song that shares a similar vibe and audio features.

Respond ONLY with a single, valid JSON object in the format:
{{"song_name": "SONG_TITLE", "artist_name": "ARTIST_NAME"}}
"""
        return prompt

    def _add_generative_song(self):
        if not self.playlist_tracks: return
        
        # Use the last 5 played tracks as the seed for the recommendation
        seed_tracks = list(self.playlist_tracks.values())[-5:]
        prompt = self._build_generative_prompt(seed_tracks)

        if not prompt:
            logger.warning("[Generative] Could not build prompt, skipping.")
            return

        try:
            response = ollama.chat(model=self.GENERATIVE_MODEL, messages=[{'role': 'user', 'content': prompt}], format="json")
            content = response['message']['content']
            
            # Robust parsing
            recommendation = json.loads(content)
            song_name = recommendation.get('song_name')
            artist_name = recommendation.get('artist_name')

            if not song_name or not artist_name:
                logger.warning(f"[Generative] LLM response missing keys. Raw: {content}")
                return

            results = self.sp.search(q=f"track:{song_name} artist:{artist_name}", limit=1, type="track")
            if not results['tracks']['items']:
                logger.warning(f"[Generative] No Spotify results for '{song_name}' by '{artist_name}'.")
                return

            track = results['tracks']['items'][0]
            
            # Avoid adding a song that's already in the main playlist
            if track['id'] in self.playlist_tracks:
                logger.info(f"[Generative] Skipping '{track['name']}' as it's already in the playlist.")
                return

            features = self._get_reccobeats_features_batch([track['id']]).get(track['id'])
            if not features: return

            embedding = self._get_embedding(self._create_embedding_prompt(track, features))
            new_track = Track(id=track['id'], name=track['name'], artist=track['artists'][0]['name'], embedding=embedding)
            
            logger.info(f"➕ [Generative] Adding to playlist and queue: '{new_track.name}' by {new_track.artist}")
            
            self.sp.playlist_add_items(self.playlist_id, [new_track.id])
            self.playback_queue.append(new_track.id)
            self.playlist_tracks[new_track.id] = new_track
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error processing generative suggestion: {e}")
            if 'response' in locals() and response:
                logger.debug(f"Raw LLM response causing error: {response.get('message', {}).get('content', 'N/A')}")
        except Exception as e:
            logger.error(f"Unexpected error in generative suggestion: {e}", exc_info=True)

    def _add_llm_suggestion_based_on_top_scores(self):
        if not self.playlist_tracks:
            return
            
        # Get top 10 tracks by score to use as a seed
        top_tracks = sorted(self.playlist_tracks.values(), key=lambda t: t.score, reverse=True)[:10]
        prompt = self._build_generative_prompt(top_tracks)

        if not prompt:
            logger.warning("[LLM Suggestion] Could not build prompt, skipping.")
            return

        try:
            response = ollama.chat(model=self.GENERATIVE_MODEL, messages=[{'role': 'user', 'content': prompt}], format="json")
            content = response['message']['content']
            recommendation = json.loads(content)
            
            song_name, artist_name = recommendation.get('song_name'), recommendation.get('artist_name')
            if not song_name or not artist_name:
                logger.warning(f"[LLM Suggestion] LLM response missing keys. Raw: {content}")
                return

            results = self.sp.search(q=f"track:{song_name} artist:{artist_name}", limit=1, type="track")
            if not results['tracks']['items']:
                logger.warning(f"[LLM Suggestion] No Spotify results for '{song_name}' by '{artist_name}'.")
                return

            track = results['tracks']['items'][0]

            if track['id'] in self.playlist_tracks:
                logger.info(f"[LLM Suggestion] Skipping '{track['name']}' as it's already in the playlist.")
                return

            features = self._get_reccobeats_features_batch([track['id']]).get(track['id'])
            if not features:
                return

            embedding = self._get_embedding(self._create_embedding_prompt(track, features))
            new_track = Track(id=track['id'], name=track['name'], artist=track['artists'][0]['name'], embedding=embedding)
            
            logger.info(f"➕ [LLM Suggestion] Adding to queue: '{new_track.name}' by {new_track.artist}")
            
            self.playback_queue.append(new_track.id)
            self.playlist_tracks[new_track.id] = new_track
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error processing LLM suggestion: {e}")
            if 'response' in locals() and response:
                logger.debug(f"Raw LLM response causing error: {response.get('message', {}).get('content', 'N/A')}")
        except Exception as e:
            logger.error(f"Unexpected error in LLM suggestion: {e}", exc_info=True)


    def _reorder_queue_after_current(self, current_id):
        # Group tracks by state: remaining (not jumped/skipped/played), jumped, skipped, full_played
        remaining, jumped, skipped, played_full = [], [], [], []
        for tid, track in self.playlist_tracks.items():
            if tid == current_id:
                continue
            # Use skip_count and score heuristic for states
            if getattr(track, 'played_full', False):
                played_full.append(tid)
            elif getattr(track, 'was_jumped', False):
                jumped.append(tid)
            elif track.skip_count > 0:
                skipped.append(tid)
            else:
                remaining.append(tid)

        def order_ids(ids):
            return sorted(ids, key=lambda t: self.playlist_tracks[t].score, reverse=True)

        new_order = order_ids(remaining) + order_ids(jumped) + order_ids(skipped) + order_ids(played_full)
        self.playback_queue = new_order
        logger.info(f"Queue reordered after current: {self.playlist_tracks[current_id].name} | {[self.playlist_tracks[tid].name for tid in self.playback_queue]}")
        # Enqueue next tracks into Spotify player queue
        self._enqueue_next_tracks()

    def _enqueue_next_tracks(self, limit: int = 20):
        """Enqueue the next tracks from the internal queue into Spotify's queue.
        Respect Spotify's hard limit and avoid duplicating items already in the queue.
        Rate-limit enqueues to avoid 429s.
        """
        try:
            # Determine remaining capacity in Spotify queue (hard limit 81)
            remaining_capacity = self.SPOTIFY_QUEUE_HARD_LIMIT
            existing_queue_ids = []
            try:
                # Spotipy has `current_user_queue` in newer versions; fall back to `queue` if present
                get_queue = getattr(self.sp, 'current_user_queue', None) or getattr(self.sp, 'queue', None)
                if callable(get_queue):
                    q = get_queue()
                    queue_items = q.get('queue', []) if isinstance(q, dict) else []
                    existing_queue_ids = [it.get('id') for it in queue_items if it and it.get('id')]
                    remaining_capacity = max(0, self.SPOTIFY_QUEUE_HARD_LIMIT - len(existing_queue_ids))
                    # Also store Spotify's next track id for adjacency detection
                    try:
                        currently_playing = q.get('currently_playing') if isinstance(q, dict) else None
                        next_id = queue_items[0].get('id') if queue_items else None
                        self._last_spotify_next_id = next_id
                    except Exception:
                        pass
            except Exception:
                # If we fail to get queue, assume empty to be safe
                remaining_capacity = self.SPOTIFY_QUEUE_HARD_LIMIT

            if remaining_capacity <= 0:
                logger.info("Spotify queue is at or above the hard limit; skipping enqueue.")
                return

            # Filter out tracks already present in Spotify queue to avoid duplicates
            desired_order = [tid for tid in self.playback_queue if tid not in existing_queue_ids]
            to_take = min(limit, remaining_capacity, len(desired_order))
            to_enqueue = desired_order[:to_take]
            enqueued = 0
            for tid in to_enqueue:
                try:
                    self.sp.add_to_queue(uri=f'spotify:track:{tid}')
                    enqueued += 1
                    # Small delay to avoid 429 rate limiting
                    time.sleep(0.2)
                except Exception as e:
                    # Break on 429 storms to retry on next loop
                    if hasattr(e, 'http_status') and e.http_status == 429:
                        logger.warning("Hit 429 while enqueuing; will retry later.")
                        break
                    logger.warning(f"Failed to enqueue track {tid}: {e}")
            if enqueued:
                logger.info(f"Enqueued {enqueued} tracks to Spotify queue.")
        except Exception as e:
            logger.warning(f"Failed to enqueue tracks to Spotify queue: {e}")