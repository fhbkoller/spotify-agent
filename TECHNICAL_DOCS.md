# Technical Documentation: Spotify Agent

## Overview
Spotify Agent is an AI-driven playlist manager and intelligent shuffler for Spotify. It leverages Spotify's API, local vector search (ChromaDB), and large language models (via Ollama) to create a dynamic, adaptive listening experience. The system tracks user interactions, updates track scores, and augments playlists with both similar and generative AI-recommended songs.

---

## Architecture

### Main Components

- **main.py**: Entry point. Handles CLI arguments, logging, and launches the `IntelligentShuffler` agent for a given playlist.
- **create_playlist.py**: Script to create a new playlist from all liked songs.
- **spotify_agent.py**: Contains the core logic, including the `IntelligentShuffler` class and the `Track` dataclass.
- **persistence.py**: Manages persistent storage using SQLite (track stats) and ChromaDB (vector embeddings).

### Data Flow
1. **Initialization**: User provides a playlist URL/ID. The agent loads playlist and library tracks, syncing metadata and embeddings from the database or Spotify API.
2. **Playback Monitoring**: The agent polls Spotify for playback state, tracking which song is playing, skipped, or finished.
3. **Score Update**: When a track is skipped or finished, its score is updated in memory and persisted to the database.
4. **Track Selection**: The next track is chosen probabilistically, weighted by score.
5. **Song Addition**: When the playlist nears its end, the agent adds new songs:
   - **Similarity Search**: Finds similar tracks using ChromaDB vector search.
   - **Generative AI**: Uses an LLM to recommend new songs based on recent history, then searches Spotify for matches.
6. **Persistence**: All track stats and embeddings are stored in SQLite and ChromaDB for fast future access.

---

## Core Classes

### `Track` (dataclass)
- `id`: Spotify track ID
- `name`: Track name
- `artist`: Artist name
- `embedding`: Numpy array (vector embedding)
- `score`: Float (preference score)
- `skip_count`: Int (number of skips)

### `IntelligentShuffler`
- **Responsibilities**:
  - Manage playlist and library tracks
  - Monitor playback and update scores
  - Add new tracks via similarity or generative AI
  - Interface with Spotify API, ChromaDB, and SQLite
- **Key Methods**:
  - `initialize()`: Loads tracks, syncs with DB, fetches missing data
  - `run()`: Main loop for playback monitoring and playlist management
  - `_update_scores()`: Adjusts track scores based on user behavior
  - `_get_next_track()`: Selects the next track to play
  - `_add_similar_song()`: Adds a similar track from the library
  - `_add_generative_song()`: Uses LLM to recommend and add a new track

### `PersistenceManager`
- **Responsibilities**:
  - Store and retrieve track metadata and embeddings
  - Update track scores and skip counts
  - Find similar tracks using vector search
- **Key Methods**:
  - `get_tracks_by_ids()`: Loads tracks and embeddings from DB
  - `save_track()`: Persists new track and embedding
  - `update_track_stats()`: Updates score and skip count
  - `find_similar_tracks()`: Returns similar tracks via ChromaDB

---

## Database Schema
- **SQLite (`spotify_agent.db`)**: Stores track ID, name, artist, score, skip count
- **ChromaDB (`chroma_db/`)**: Stores vector embeddings for each track

---

## External Services
- **Spotify Web API**: For playlist, track, and playback management
- **Ollama**: For local LLM and embedding generation
- **ReccoBeats API**: For audio feature enrichment

---

## Extending the System
- Swap out embedding or LLM models by changing constants in `IntelligentShuffler`
- Add new recommendation strategies by extending `_add_similar_song` or `_add_generative_song`
- Integrate with other music services by adapting the API interface

---

## Troubleshooting
- Check `spotify_agent.log` for detailed logs
- Ensure Spotify API credentials are set in `.env`
- Make sure Ollama is running and models are available
