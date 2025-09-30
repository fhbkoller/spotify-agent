# spotify_agent.py

import os
import time
import ollama
import spotipy
import numpy as np
import json
from tqdm import tqdm
from spotipy.oauth2 import SpotifyOAuth
from sklearn.metrics.pairwise import cosine_similarity

# --- Mentorship Note ---
# In Python, we often use 'dataclasses' to create simple classes for holding data.
# It's similar to Kotlin's 'data class' or a Java record. It automatically
# gives you methods like __init__, __repr__, etc., making the code cleaner.
from dataclasses import dataclass, field

@dataclass
class Track:
    """A simple data structure to hold all relevant information about a track."""
    id: str
    name: str
    artist: str
    embedding: np.ndarray = field(repr=False) # The 'vibe fingerprint'
    score: float = 1.0 # The dynamic preference score
    skip_count: int = 0

class IntelligentShuffler:
    """
    An agent that dynamically manages a Spotify playlist based on user actions.
    """
    # --- Constants for score adjustments ---
    SKIP_PENALTY = 0.5
    FINISH_BONUS = 0.1
    JUMP_BONUS = 0.3
    POLLING_INTERVAL_SECONDS = 5 # Check Spotify every 5 seconds

    def __init__(self, playlist_id: str):
        self._setup_spotify_client()
        self.playlist_id = playlist_id
        self.user_id = self.sp.me()['id']

        self.playlist_tracks: dict[str, Track] = {}
        self.library_tracks: dict[str, Track] = {}

        self.current_track_id: str | None = None
        self.last_track_id: str | None = None
        self.last_progress_ms: int = 0

    def _setup_spotify_client(self):
        """Authenticates with the Spotify API."""
        scope = "user-read-playback-state user-modify-playback-state playlist-modify-public user-library-read"
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    # --- Mentorship Note ---
    # This is an f-string (formatted string literal). It's the modern, idiomatic
    # way to embed expressions inside strings in Python. Think of it like Kotlin's
    # string templates (e.g., "Name: $name") - much cleaner than older methods.
    def _create_embedding_prompt(self, track_info, features) -> str:
        """Creates a detailed text description of a track for the embedding model."""
        return (
            f"Track: {track_info['name']} by {track_info['artists'][0]['name']}. "
            f"Album: {track_info['album']['name']}. "
            f"Audio features: "
            f"danceability is {features['danceability']:.2f}, "
            f"energy is {features['energy']:.2f}, "
            f"valence (positivity) is {features['valence']:.2f}, "
            f"tempo is {features['tempo']:.0f} BPM."
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates a vector embedding for a given text using a local LLM."""
        response = ollama.embeddings(model='mxbai-embed-large', prompt=text)
        return np.array(response['embedding'])

    def _fetch_and_embed_tracks(self, get_tracks_func, description: str) -> dict[str, Track]:
        """A generic function to fetch tracks and generate their embeddings."""
        print(f"\nFetching and processing tracks from {description}...")
        tracks_map = {}
        
        results = get_tracks_func()
        items = results['items']
        while results['next']:
            results = self.sp.next(results)
            items.extend(results['items'])

        track_infos = [item['track'] for item in items if item.get('track')]
        track_ids = [t['id'] for t in track_infos if t and t.get('id')]
        
        # Batch fetch audio features for efficiency
        for i in tqdm(range(0, len(track_ids), 100), desc=f"Embedding {description}"):
            batch_ids = track_ids[i:i+100]
            batch_infos = track_infos[i:i+100]
            try:
                features_list = self.sp.audio_features(batch_ids)
                for track_info, features in zip(batch_infos, features_list):
                    if not track_info or not features:
                        continue
                    prompt = self._create_embedding_prompt(track_info, features)
                    embedding = self._get_embedding(prompt)
                    tracks_map[track_info['id']] = Track(
                        id=track_info['id'],
                        name=track_info['name'],
                        artist=track_info['artists'][0]['name'],
                        embedding=embedding,
                    )
            except Exception as e:
                print(f"Error processing batch: {e}")

        return tracks_map

    def initialize(self):
        """Loads all necessary data and prepares the agent for operation."""
        print("Initializing Intelligent Shuffle Agent...")
        self.playlist_tracks = self._fetch_and_embed_tracks(
            lambda: self.sp.playlist_items(self.playlist_id), "target playlist"
        )
        self.library_tracks = self._fetch_and_embed_tracks(
            lambda: self.sp.current_user_saved_tracks(limit=50), "your 'Liked Songs'"
        )
        print("\nInitialization complete. Starting playback monitoring.")
        
        # Initial shuffle of the playlist
        self.sp.shuffle(state=True)
        # Go to the next track to ensure shuffle takes effect
        self.sp.next_track()


    def run(self):
        """The main agent loop for monitoring and reacting."""
        if not self.playlist_tracks:
            print("Playlist is empty or could not be loaded. Exiting.")
            return

        while True:
            try:
                playback_state = self.sp.current_playback()

                if playback_state and playback_state['is_playing'] and playback_state['item']:
                    self._process_playback_state(playback_state)
                else:
                    print("Playback paused or stopped. Waiting...", end="\r")

                time.sleep(self.POLLING_INTERVAL_SECONDS)

            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                time.sleep(30) # Wait longer after an error

    def _process_playback_state(self, state):
        """Analyzes the playback state to detect user actions."""
        new_track_id = state['item']['id']
        progress_ms = state['progress_ms']
        duration_ms = state['item']['duration_ms']

        self.current_track_id = new_track_id

        # Detect a song change
        if self.last_track_id and self.last_track_id != self.current_track_id:
            # Check if the last song was finished or skipped
            if self.last_progress_ms / duration_ms > 0.95: # Finished
                print(f"\n✅ Finished: '{self.playlist_tracks[self.last_track_id].name}'")
                self._update_scores(self.last_track_id, 'finish')
            else: # Skipped
                print(f"\n⏭️ Skipped: '{self.playlist_tracks[self.last_track_id].name}'")
                self._update_scores(self.last_track_id, 'skip')
            
            self._reorder_and_update_playlist()

        self.last_track_id = self.current_track_id
        self.last_progress_ms = progress_ms

    def _update_scores(self, track_id: str, action: str):
        """Updates the preference scores based on the user action."""
        if track_id not in self.playlist_tracks:
            return

        track = self.playlist_tracks[track_id]
        if action == 'skip':
            track.score *= self.SKIP_PENALTY
            track.skip_count += 1
            print(f"   📉 Score for '{track.name}' decreased to {track.score:.2f}")
        elif action == 'finish':
            track.score += self.FINISH_BONUS
            print(f"   📈 Score for '{track.name}' increased to {track.score:.2f}")
        
        # We could add a 'jump' action here, but detecting it reliably via polling is complex.
        # We'll focus on skip/finish for V1.

    def _reorder_and_update_playlist(self):
        """Re-ranks, handles removals/additions, and updates the Spotify queue."""
        
        # Handle removals for tracks skipped too many times
        tracks_to_remove = [tid for tid, track in self.playlist_tracks.items() if track.skip_count >= 2]
        if tracks_to_remove:
            for tid in tracks_to_remove:
                track_name = self.playlist_tracks[tid].name
                print(f"   🚫 Removing '{track_name}' from playlist (skipped {self.playlist_tracks[tid].skip_count} times).")
                self.sp.remove_playlist_items(self.playlist_id, [tid])
                del self.playlist_tracks[tid]
                self._add_new_song_generative()

        # Get the currently playing track ID to keep it at the top
        now_playing_id = self.sp.current_playback()['item']['id']

        # Sort remaining tracks by score (descending)
        remaining_tracks = [t for t in self.playlist_tracks.values() if t.id != now_playing_id]
        sorted_tracks = sorted(remaining_tracks, key=lambda t: t.score, reverse=True)
        
        print("   🎶 Reordering playlist based on new scores...")
        # Spotify's API to reorder is a bit tricky. The easiest way to influence the queue
        # is to remove all tracks and add them back in the new order.
        # A less disruptive way is to add tracks to the queue one by one. Let's do that.
        for track in sorted_tracks:
            try:
                # This adds the track to the *end* of the queue.
                self.sp.add_to_queue(track.id)
                print(f"      - Queued '{track.name}' (Score: {track.score:.2f})")
            except Exception as e:
                # Might fail if the queue is full, etc.
                pass

    def _add_new_song_retrieval(self):
        """Finds and adds a new song from the library using vector similarity (Retrieval)."""
        # ... (The exact code from our previous _add_new_song method goes here)
        # I'll paste it here for clarity.
        if not self.library_tracks or not self.playlist_tracks:
            return

        top_tracks = sorted(self.playlist_tracks.values(), key=lambda t: t.score, reverse=True)[:3]
        if not top_tracks:
            return
        
        target_embedding = np.mean([t.embedding for t in top_tracks], axis=0).reshape(1, -1)
        library_candidates = [t for t in self.library_tracks.values() if t.id not in self.playlist_tracks]
        
        if not library_candidates:
            print("   ⚠️ No new songs in library to add via retrieval.")
            return

        candidate_embeddings = np.array([t.embedding for t in library_candidates])
        similarities = cosine_similarity(target_embedding, candidate_embeddings)
        best_match_index = np.argmax(similarities)
        new_track = library_candidates[best_match_index]

        print(f"   ➕ [Retrieval] Adding: '{new_track.name}' by {new_track.artist}")
        self.sp.add_to_playlist(self.playlist_id, [new_track.id])
        self.playlist_tracks[new_track.id] = new_track


    # --- And now, we add your proposed generative method ---
    def _add_new_song_generative(self):
        """Asks a generative LLM to recommend a brand new song (Generative Discovery)."""
        print("   🧠 Engaging generative model for a new song recommendation...")
        if not self.playlist_tracks:
            return

        top_tracks = sorted(self.playlist_tracks.values(), key=lambda t: t.score, reverse=True)[:5]
        if not top_tracks:
            return

        top_tracks_str = "\n".join([f"- '{t.name}' by {t.artist}" for t in top_tracks])
        artist_list_str = ", ".join([f"'{t.artist}'" for t in top_tracks])

        # --- Mentorship Note: This is Prompt Engineering ---
        # Crafting a good prompt is an art. We give it a role, context, a clear task,
        # constraints, and a strict output format. This heavily influences the quality
        # of the model's response.
        prompt = f"""
        You are a world-class music discovery expert with an encyclopedic knowledge of music.
        Your task is to recommend ONE song that fits the vibe of a user's current playlist.

        The current top songs in the playlist are:
        {top_tracks_str}

        Based on this vibe, recommend a new song.

        CONSTRAINTS:
        1. The song MUST be a real song available on Spotify.
        2. The song's artist MUST NOT be one of these: {artist_list_str}.
        3. You must respond ONLY with a single JSON object, with no other text before or after it.

        JSON FORMAT:
        {{
        "song_name": "...",
        "artist_name": "..."
        }}
        """

        try:
            response = ollama.chat(
                model='qwen:7b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7} # A little creativity is good
            )
            content = response['message']['content']
            
            # --- Validation Step 1: Parse the JSON ---
            recommendation = json.loads(content)
            song_name = recommendation['song_name']
            artist_name = recommendation['artist_name']

            print(f"   🤖 LLM suggested: '{song_name}' by {artist_name}")

            # --- Validation Step 2: Check Spotify ---
            # This is CRITICAL to avoid hallucinations
            search_result = self.sp.search(q=f"track:{song_name} artist:{artist_name}", type="track", limit=1)
            
            if not search_result['tracks']['items']:
                print(f"   ⚠️ LLM suggestion '{song_name}' not found on Spotify. Skipping.")
                return

            new_track_info = search_result['tracks']['items'][0]
            new_track_id = new_track_info['id']
            
            if new_track_id in self.playlist_tracks:
                print(f"   ℹ️ LLM suggested a song already in the playlist. Skipping.")
                return

            # We need to create an embedding for this new track to add it to our state
            features = self.sp.audio_features([new_track_id])[0]
            embedding_prompt = self._create_embedding_prompt(new_track_info, features)
            embedding = self._get_embedding(embedding_prompt)
            
            new_track = Track(
                id=new_track_id,
                name=new_track_info['name'],
                artist=new_track_info['artists'][0]['name'],
                embedding=embedding
            )

            print(f"   ➕ [Generative] Adding: '{new_track.name}' by {new_track.artist}")
            self.sp.add_to_playlist(self.playlist_id, [new_track.id])
            self.playlist_tracks[new_track.id] = new_track

        except Exception as e:
            print(f"   ❌ Error during generative recommendation: {e}")