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

### --- MENTORSHIP NOTE: REMOVED SCORE-RELATED COLUMNS --- ###
# We no longer persist scores, so these columns are not needed in the main table definition
# for new tables. The code will still work with old tables, it just won't use the columns.
songs_table = Table(
    "songs",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("artist", String, nullable=False),
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
        if found_ids:
            chunk_size = 100
            for i in range(0, len(found_ids), chunk_size):
                batch_ids = found_ids[i:i+chunk_size]
                try:
                    embeddings_result = self.embedding_collection.get(ids=batch_ids, include=["embeddings"])
                    ids_with_embeddings = embeddings_result.get('ids', [])
                    embeddings = embeddings_result.get('embeddings', [])
                    for sid, embedding in zip(ids_with_embeddings, embeddings):
                        if sid in found_tracks and embedding is not None:
                            found_tracks[sid]['embedding'] = np.array(embedding)
                except Exception as e:
                    logger.error(f"Error fetching embeddings for batch: {e}")


        missing_ids = [sid for sid in spotify_ids if sid not in found_tracks]
        
        logger.debug(f"DB lookup: Found {len(found_tracks)} tracks, missing {len(missing_ids)} tracks.")
        return found_tracks, missing_ids

    def save_track(self, track_data: dict):
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        stmt = sqlite_insert(songs_table).values(
            id=track_data["id"],
            name=track_data["name"],
            artist=track_data["artist"],
        ).on_conflict_do_nothing()

        with self.db_engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

        if 'embedding' in track_data and track_data['embedding'] is not None:
            self.save_embedding(track_data["id"], track_data["embedding"])
        logger.debug(f"Saved track '{track_data['name']}' to persistence.")

    def save_embedding(self, track_id: str, embedding: np.ndarray):
        try:
            self.embedding_collection.upsert(
                ids=[track_id],
                embeddings=[embedding.tolist()],
            )
        except Exception as e:
                logger.error(f"Failed to write embedding for {track_id}: {e}")
        
    def find_similar_tracks(self, vector: np.ndarray, n_results: int = 5, exclude_ids: list = None, include_distances: bool = False) -> list:
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
        
        final_ids, final_distances = [], []
        if results and results.get('ids') and results['ids'][0]:
            for i, sid in enumerate(results['ids'][0]):
                if sid not in exclude_ids:
                    final_ids.append(sid)
                    if results.get('distances') and results['distances'][0]:
                        final_distances.append(results['distances'][0][i])
        
        if include_distances:
            return list(zip(final_ids, final_distances))[:n_results]
        else:
            return final_ids[:n_results]