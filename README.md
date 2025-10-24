# Spotify Agent

Spotify Agent is an AI-powered playlist manager and intelligent shuffler for Spotify. It uses machine learning and large language models to enhance your listening experience by dynamically managing playlists, recommending new tracks, and learning from your listening behavior.

## Features

- **Intelligent Shuffle:** Learns your preferences and adapts playlist order based on skips, finishes, and track similarity.
- **AI Recommendations:** Adds new songs using both similarity search (vector embeddings) and generative AI suggestions (via Ollama).
- **Persistent Memory:** Remembers track stats and embeddings using SQLite and ChromaDB.
- **Full Liked Songs Import:** Easily create a playlist from all your liked songs.
- **Robust Logging:** Detailed logs for debugging and monitoring (saved to `data/spotify_agent.log`).

## Setup

### Prerequisites

- Python 3.9+
- Spotify Developer account (for API credentials)
- [Ollama](https://ollama.com/) (for local LLM and embedding models, required for AI features)

### Installation

1. Clone this repository:
    ```sh
    git clone <repo-url>
    cd spotify-agent
    ```

2. (Recommended) Create and activate a virtual environment:
    ```sh
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. Install dependencies:
    ```sh
    pip install spotipy tqdm python-dotenv sqlalchemy chromadb numpy requests ollama
    ```

4. Set up your Spotify API credentials in a `.env` file:
    ```env
    SPOTIPY_CLIENT_ID=your_client_id
    SPOTIPY_CLIENT_SECRET=your_client_secret
    SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
    ```

5. Ensure Ollama is running and required models are available.

## Usage

### 1. Create a Playlist from Liked Songs

Run:
```sh
python create_playlist.py
```
Follow the prompts to create a new playlist containing all your liked songs. Copy the resulting playlist URL or ID.

### 2. Run the Intelligent Shuffler

```sh
python main.py <playlist_url_or_id>
```
The agent will:
- Monitor playback
- Adaptively reorder and update the playlist based on listening behavior
- Add new tracks based on user preferences and AI recommendations

### Logs

All logs are saved to `data/spotify_agent.log`.

## Project Structure

The project follows clean architecture principles with clear separation of concerns:

- `src/core/` — Core application logic (agent, models, main entry point)
- `src/infrastructure/` — Database and external service management
- `src/services/` — Business logic services (ready for future expansion)
- `src/utils/` — Utilities and logging configuration
- `ml/` — Machine learning components (training, testing, data processing)
- `scripts/` — Utility scripts for specific tasks
- `data/` — All data storage (models, databases, logs, training data)
- `main.py` — Main application entry point
- `create_playlist.py` — Playlist creation script

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT
