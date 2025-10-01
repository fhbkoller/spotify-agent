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

logger = logging.getLogger(__name__)

DB_FILE = "spotify_agent.db"
CHROMA_PATH = "chroma_db"
EMBEDDING_COLLECTION_NAME = "song_embeddings"

metadata = MetaData()

### --- FIX: Renamed 'spotify_id' to 'id' to match the Track object --- ###
songs_table = Table(
    "songs",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("artist", String, nullable=False),
    Column("score", Float, default=1.0),
    Column("skip_count", Integer, default=0),
)


class PersistenceManager:
    def __init__(self):
        self.db_engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)
        metadata.create_all(self.db_engine)
        logger.info(f"Initialized and connected to SQLite database at '{DB_FILE}'")

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedding_collection = self.chroma_client.get_or_create_collection(name=EMBEDDING_COLLECTION_NAME)
        logger.info(f"Initialized and connected to ChromaDB at '{CHROMA_PATH}'")

    def get_tracks_by_ids(self, spotify_ids: list[str]) -> tuple[dict, list]:
        if not spotify_ids:
            return {}, []

        found_tracks = {}
        
        stmt = select(songs_table).where(songs_table.c.id.in_(spotify_ids))
        with self.db_engine.connect() as conn:
            for row in conn.execute(stmt):
                row_dict = row._asdict()
                found_tracks[row_dict['id']] = row_dict

        found_ids = list(found_tracks.keys())
        # Fetch embeddings in chunks to avoid backend limits
        if found_ids:
            chunk_size = 100
            for i in range(0, len(found_ids), chunk_size):
                batch_ids = found_ids[i:i+chunk_size]
                embeddings_result = self.embedding_collection.get(ids=batch_ids, include=["embeddings"])
                ids_with_embeddings = embeddings_result.get('ids')
                if ids_with_embeddings is None:
                    ids_with_embeddings = []
                embeddings = embeddings_result.get('embeddings')
                if embeddings is None:
                    embeddings = []
                for sid, embedding in zip(ids_with_embeddings, embeddings):
                    if sid in found_tracks and embedding is not None:
                        found_tracks[sid]['embedding'] = np.array(embedding)

        missing_ids = [sid for sid in spotify_ids if sid not in found_tracks]
        
        logger.debug(f"DB lookup: Found {len(found_tracks)} tracks, missing {len(missing_ids)} tracks.")
        return found_tracks, missing_ids

    def save_track(self, track_data: dict):
        """Save or upsert the song row. Embedding is optional and saved separately if provided."""
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        stmt = sqlite_insert(songs_table).values(
            id=track_data["id"],
            name=track_data["name"],
            artist=track_data["artist"],
            score=1.0,
            skip_count=0,
        ).on_conflict_do_nothing()

        with self.db_engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

        if 'embedding' in track_data and track_data['embedding'] is not None:
            self.save_embedding(track_data["id"], track_data["embedding"])
        logger.debug(f"Saved track row '{track_data['name']}' to SQLite and optional embedding to Chroma.")

    def save_embedding(self, track_id: str, embedding: np.ndarray):
        try:
            self.embedding_collection.add(
                ids=[track_id],
                embeddings=[embedding.tolist()],
            )
        except Exception:
            try:
                # Fallback to update if already exists
                self.embedding_collection.update(
                    ids=[track_id],
                    embeddings=[embedding.tolist()],
                )
            except Exception as e:
                logger.error(f"Failed to write embedding for {track_id}: {e}")

    def update_track_stats(self, track_id: str, new_score: float, new_skip_count: int):
        stmt = (
            update(songs_table)
            .where(songs_table.c.id == track_id)
            .values(score=new_score, skip_count=new_skip_count)
        )
        with self.db_engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()
        logger.debug(f"Updated stats for track {track_id}: score={new_score:.2f}, skips={new_skip_count}")
        
    def find_similar_tracks(self, vector: np.ndarray, n_results: int = 5, exclude_ids: list = None) -> list:
        if exclude_ids is None:
            exclude_ids = []
        
        results_to_fetch = n_results + len(exclude_ids)
        if self.embedding_collection.count() < results_to_fetch:
            results_to_fetch = self.embedding_collection.count()
            
        if results_to_fetch == 0: return []

        results = self.embedding_collection.query(
            query_embeddings=[vector.tolist()],
            n_results=results_to_fetch,
        )
        
        similar_ids = [sid for sid in results['ids'][0] if sid not in exclude_ids]
        
        return similar_ids[:n_results]