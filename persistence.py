import logging
import chromadb
import numpy as np
from sqlalchemy import (
    create_engine,
    Table,
    Column,
    MetaData,
    String,
    Float,
    Integer,
    select,
    insert,
    update,
)

# Use the standard __name__ to get the logger for this module
logger = logging.getLogger(__name__)

# --- Database Configuration ---
DB_FILE = "spotify_agent.db"
CHROMA_PATH = "chroma_db"
EMBEDDING_COLLECTION_NAME = "song_embeddings"

# --- Relational Schema Definition (using SQLAlchemy) ---
# The MetaData object holds all the information about our database schema.
metadata = MetaData()

# We define our 'songs' table with its columns and data types.
songs_table = Table(
    "songs",
    metadata,
    Column("spotify_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("artist", String, nullable=False),
    Column("score", Float, default=1.0),
    Column("skip_count", Integer, default=0),
    # We could add more columns here like 'album', 'duration_ms', etc.
)


class PersistenceManager:
    """
    Handles all database interactions for both relational (SQLite)
    and vector (ChromaDB) storage.
    """

    def __init__(self):
        # --- SQLite Setup ---
        # The engine is the entry point to our database.
        # `echo=False` prevents it from logging every SQL statement.
        self.db_engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)
        # Create the 'songs' table if it doesn't already exist.
        metadata.create_all(self.db_engine)
        logger.info(f"Initialized and connected to SQLite database at '{DB_FILE}'")

        # --- ChromaDB Setup ---
        # This initializes ChromaDB, storing its data in the specified directory.
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Get or create the collection that will store our song embeddings.
        self.embedding_collection = self.chroma_client.get_or_create_collection(
            name=EMBEDDING_COLLECTION_NAME
        )
        logger.info(f"Initialized and connected to ChromaDB at '{CHROMA_PATH}'")

    def get_tracks_by_ids(self, spotify_ids: list[str]) -> tuple[dict, list]:
        """
        Fetches tracks that already exist in our local databases.
        Returns a tuple containing:
        1. A dictionary of found tracks with their relational data.
        2. A list of spotify_ids that were NOT found.
        """
        if not spotify_ids:
            return {}, []

        found_tracks = {}
        
        # --- Query SQLite for metadata and stats ---
        stmt = select(songs_table).where(songs_table.c.spotify_id.in_(spotify_ids))
        with self.db_engine.connect() as conn:
            for row in conn.execute(stmt):
                # Using ._asdict() requires SQLAlchemy 2.0+ style result rows
                row_dict = row._asdict()
                found_tracks[row_dict['spotify_id']] = row_dict

        # --- Query ChromaDB for embeddings ---
        # Note: ChromaDB can take a list of IDs and returns what it finds.
        found_ids = list(found_tracks.keys())
        if found_ids:
            embeddings_result = self.embedding_collection.get(ids=found_ids, include=["embeddings"])
            
            # Add the embeddings to our found_tracks dictionary
            for sid, embedding in zip(embeddings_result['ids'], embeddings_result['embeddings']):
                if sid in found_tracks:
                    found_tracks[sid]['embedding'] = np.array(embedding)

        # Determine which tracks we couldn't find locally
        missing_ids = [sid for sid in spotify_ids if sid not in found_tracks]
        
        logger.debug(f"DB lookup: Found {len(found_tracks)} tracks, missing {len(missing_ids)} tracks.")
        return found_tracks, missing_ids


    def save_track(self, track_data: dict):
        """
        Saves a new track's complete data to both databases.
        Expects a dictionary with keys: spotify_id, name, artist, embedding.
        """
        if 'embedding' not in track_data:
            logger.error(f"Cannot save track {track_data.get('spotify_id')}: Missing embedding.")
            return

        # --- Save relational data to SQLite ---
        stmt = insert(songs_table).values(
            spotify_id=track_data["spotify_id"],
            name=track_data["name"],
            artist=track_data["artist"],
            score=1.0,  # New tracks start with a default score
            skip_count=0,
        )
        # Use on_conflict_do_nothing to prevent errors if a track is somehow fetched twice.
        # This requires SQLite 3.24+
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        stmt = sqlite_insert(songs_table).values(
            spotify_id=track_data["spotify_id"],
            name=track_data["name"],
            artist=track_data["artist"],
        ).on_conflict_do_nothing()


        with self.db_engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

        # --- Save vector embedding to ChromaDB ---
        self.embedding_collection.add(
            ids=[track_data["spotify_id"]],
            embeddings=[track_data["embedding"].tolist()], # Chroma needs a list of lists
            # We could also store metadata here, e.g., {'name': track_data['name']}
        )
        logger.debug(f"Saved new track '{track_data['name']}' to local databases.")


    def update_track_stats(self, spotify_id: str, new_score: float, new_skip_count: int):
        """
        Updates the score and skip_count for a specific track in SQLite.
        """
        stmt = (
            update(songs_table)
            .where(songs_table.c.spotify_id == spotify_id)
            .values(score=new_score, skip_count=new_skip_count)
        )
        with self.db_engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()
        logger.debug(f"Updated stats for track {spotify_id}: score={new_score:.2f}, skips={new_skip_count}")
        
    def find_similar_tracks(self, vector: np.ndarray, n_results: int = 5, exclude_ids: list = None) -> list:
        """
        Finds the most similar tracks from the vector database.
        Optionally excludes a list of IDs from the search results.
        """
        if exclude_ids is None:
            exclude_ids = []
        
        # Build the 'where' clause for ChromaDB filtering
        # This is a bit simplistic; more complex logic is possible
        # For now, we filter in Python after the query.
        
        results = self.embedding_collection.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results + len(exclude_ids), # Fetch more to account for filtering
        )
        
        # Filter out excluded IDs
        similar_ids = [sid for sid in results['ids'][0] if sid not in exclude_ids]
        
        # Return the top n_results
        return similar_ids[:n_results]