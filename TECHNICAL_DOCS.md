## Overview

Spotify Agent is an AI-driven playlist manager and intelligent shuffler for Spotify. It leverages Spotify's API, local vector search (ChromaDB), and large language models (via Ollama) to create a dynamic, adaptive listening experience. The system tracks user interactions, updates track scores, and augments playlists with both similar and generative AI-recommended songs.

---

### Main Components

- **main.py:** Entry point. Handles CLI arguments, logging, and launches the `IntelligentShuffler` agent for a given playlist.
- **create_playlist.py:** Script to create a new playlist from all liked songs.
- **spotify_agent.py:** Contains the core logic, including the `IntelligentShuffler` class and the `Track` dataclass.
- **persistence.py:** Manages persistent storage using SQLite (track stats) and ChromaDB (vector embeddings).

---

### Data Flow

1. User launches the agent with a playlist.
2. Agent loads and syncs playlist tracks (fetches missing metadata and embeddings).
3. Tracks are managed in-memory and updated based on user behavior (skips, plays).
4. Embeddings are used for similarity-based recommendations via ChromaDB.
5. Generative AI (Ollama) suggests new tracks based on preferences and context.

---

### `Track` (dataclass)

- `id`: Spotify track ID
- `name`: Track name
- `artist`: Artist name
- `embedding`: Numpy array (vector embedding)
- `score`: Float (preference score, session-based)
- `skip_count`: Int (number of skips)
- `play_count`: Int (number of plays)
- `last_event`: Last event type (`fresh`, `skipped`, `played`)
- `is_new`: Boolean (if track was just added)

### `IntelligentShuffler`

**Responsibilities:**
- Manage playlist and library tracks
- Monitor playback and update scores
- Add new tracks via similarity or generative AI
- Interface with Spotify API, ChromaDB, and SQLite

**Key Methods:**
- `initialize()`: Loads tracks, syncs with DB, fetches missing data
- `run()`: Main loop for playback monitoring and playlist management
- `_update_scores()`: Adjusts track scores based on user behavior
- `_get_next_track()`: Selects the next track to play
- `_add_similar_song()`: Adds a similar track from the library
- `_add_generative_song()`: Uses LLM to recommend and add a new track

### `PersistenceManager`

- Handles saving/loading of track metadata, embeddings, and playlist state.
- Uses SQLite for track metadata, ChromaDB for embeddings.

---

## Database Schema

- **SQLite (`spotify_agent.db`)**: Stores track ID, name, artist
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
