# Spotify Agent

Spotify Agent is an AI-powered playlist manager and intelligent shuffler for Spotify. It uses machine learning and large language models to enhance your listening experience by dynamically managing playlists, recommending new tracks, and learning from your listening behavior.

## Features
- **Intelligent Shuffle**: Learns your preferences and adapts playlist order based on skips, finishes, and track similarity.
- **AI Recommendations**: Adds new songs using both similarity search and generative AI suggestions.
- **Persistent Memory**: Remembers track stats and embeddings using SQLite and ChromaDB.
- **Full Liked Songs Import**: Easily create a playlist from all your liked songs.
- **Robust Logging**: Detailed logs for debugging and monitoring.

## Setup

### Prerequisites
- Python 3.9+
- Spotify Developer account (for API credentials)
- [Ollama](https://ollama.com/) (for local LLM and embedding models)

### Installation
1. Clone this repository:
   ```sh
   git clone <repo-url>
   cd spotify-agent
   ```
2. Create and activate a virtual environment (optional but recommended):
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
- Adaptively reorder and update the playlist
- Add new tracks based on your listening behavior and AI recommendations

### Logs
All logs are saved to `spotify_agent.log`.

## Project Structure
- `main.py` — Entry point for the intelligent shuffler
- `create_playlist.py` — Script to create a playlist from all liked songs
- `spotify_agent.py` — Core logic for the agent and shuffling
- `persistence.py` — Database and embedding management
- `chroma_db/` — ChromaDB vector store
- `spotify_agent.db` — SQLite database for track stats

## License
MIT
