# ml/data_pipeline/triplet_miners.py

import pandas as pd
import random
from typing import Set, Dict, Optional

# --- Internal Knowledge: Vibe Mappings ---
# We now use the full, categorized tag names
VIBE_CONTRASTS = {
    'mood::sad': {'mood::happy', 'mood::joyful', 'mood::energetic', 'mood::party'},
    'mood::happy': {'mood::sad', 'mood::melancholic', 'mood::dark'},
    'mood::calm': {'mood::energetic', 'mood::party', 'mood::aggressive'},
    'mood::energetic': {'mood::calm', 'mood::relaxed', 'mood::ambient'},
    'mood::aggressive': {'mood::gentle', 'mood::calm', 'mood::ambient'},
}

VIBE_ADJACENCIES = {
    'mood::sad': {'mood::melancholic', 'mood::dark', 'mood::dramatic'},
    'mood::happy': {'mood::joyful', 'mood::bright'},
    'mood::calm': {'mood::relaxed', 'mood::ambient', 'mood::peaceful'},
    'mood::energetic': {'mood::party', 'mood::driving'},
}

# --- Internal Knowledge: Genre Mappings ---
# We now use the full, categorized tag names
GENRE_HIERARCHY = {
    'genre::rock': {'genre::hardrock', 'genre::classicrock', 'genre::punk'},
    'genre::electronic': {'genre::techno', 'genre::house', 'genre::trance', 'genre::ambient'},
    'genre::pop': {'genre::dancepop', 'genre::synthpop'},
    'genre::classical': {'genre::orchestral', 'genre::chamber', 'genre::contemporaryclassical'},
    'genre::hiphop': {'genre::rap', 'genre::trap'},
}


class BaseTripletMiner:
    """Base class for all miners."""
    def __init__(self, inverted_index: dict, tags_onehot: "pd.DataFrame"):
        self.inverted_index = inverted_index
        self.tags_onehot = tags_onehot
        
        if not inverted_index or not isinstance(tags_onehot, pd.DataFrame):
            raise ValueError("Miner initialized with invalid lookup data.")
            
        self.all_track_ids = set(self.tags_onehot.index)

    def mine(self, anchor_id: str, anchor_tags: set) -> tuple | None:
        """
        Main method to find a (positive_id, negative_id) pair.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_random_track_from_set(self, track_set: set, exclude_id: str = None) -> str | None:
        """Helper to safely get a random track ID from a set."""
        if exclude_id:
            track_set.discard(exclude_id)
        
        if not track_set:
            return None
        
        return random.choice(list(track_set))

    def _get_tracks_by_tag(self, tag: str) -> set:
        """Helper to safely query the inverted index."""
        return self.inverted_index.get(tag, set()).copy()

    def _find_negative_by_exclusion(self, anchor_tags: set) -> set:
        """Finds all tracks that share *no* tags with the anchor."""
        # Find all tracks that share AT LEAST ONE tag
        related_tracks = set()
        for tag in anchor_tags:
            related_tracks.update(self._get_tracks_by_tag(tag))
        
        # The negative pool is all tracks MINUS the related tracks
        negative_pool = self.all_track_ids - related_tracks
        return negative_pool


class SimpleTagMiner(BaseTripletMiner):
    """
    A simple, robust miner to guarantee triplets are found.
    - Positive: *Any* track that shares *at least one* tag.
    - Negative: *Any* track that shares *zero* tags.
    """
    def __init__(self, inverted_index: dict, tags_onehot: "pd.DataFrame"):
        super().__init__(inverted_index, tags_onehot)
        print("SimpleTagMiner initialized (robust fallback).")

    def mine(self, anchor_id: str, anchor_tags: set) -> tuple | None:
        
        # 1. Find Positive (Easy)
        positive_pool = set()
        for tag in anchor_tags:
            positive_pool.update(self._get_tracks_by_tag(tag))
        
        positive_id = self._get_random_track_from_set(positive_pool, exclude_id=anchor_id)
        if not positive_id:
            return None # No other track shares any tag

        # 2. Find Negative (Easy)
        negative_pool = self._find_negative_by_exclusion(anchor_tags)
        negative_id = self._get_random_track_from_set(negative_pool, exclude_id=anchor_id)
        
        if not negative_id:
            return None # No track is a perfect negative

        return positive_id, negative_id


class GenreTripletMiner(BaseTripletMiner):
    """ Mines triplets based on genre similarity. """
    
    def __init__(self, inverted_index: dict, tags_onehot: "pd.DataFrame"):
        super().__init__(inverted_index, tags_onehot)
        # Pre-compute lookups for this miner
        self.tag_to_parent_genre = {}
        self.all_parent_genres = set(GENRE_HIERARCHY.keys())
        for parent, children in GENRE_HIERARCHY.items():
            for child in children:
                self.tag_to_parent_genre[child] = parent
        
        # All known sub-genres
        self.all_sub_genres = set(self.tag_to_parent_genre.keys())
        print(f"GenreTripletMiner initialized with {len(self.all_parent_genres)} parent genres.")

    def mine(self, anchor_id: str, anchor_tags: set) -> tuple | None:
        
        # 1. Find anchor's sub-genres
        anchor_sub_genres = anchor_tags.intersection(self.all_sub_genres)
        if not anchor_sub_genres:
            return None # Anchor has no mineable sub-genre
            
        anchor_genre_tag = random.choice(list(anchor_sub_genres))
        
        # 2. Find Positive (Easy)
        positive_pool = self._get_tracks_by_tag(anchor_genre_tag)
        positive_id = self._get_random_track_from_set(positive_pool, exclude_id=anchor_id)
        if not positive_id:
            return None # No other tracks in this sub-genre

        # 3. Find Negative (Easy)
        anchor_parent_genre = self.tag_to_parent_genre[anchor_genre_tag]
        other_parent_genres = self.all_parent_genres - {anchor_parent_genre}
        if not other_parent_genres:
            return None 
            
        negative_parent_tag = random.choice(list(other_parent_genres))
        negative_sub_genres = GENRE_HIERARCHY[negative_parent_tag]
        if not negative_sub_genres:
            return None
            
        negative_genre_tag = random.choice(list(negative_sub_genres))
        
        negative_pool = self._get_tracks_by_tag(negative_genre_tag)
        negative_id = self._get_random_track_from_set(negative_pool, exclude_id=anchor_id)
        
        if not negative_id:
            return None

        return positive_id, negative_id


class VibeTripletMiner(BaseTripletMiner):
    """ Mines triplets based on mood/vibe similarity. """
    
    def __init__(self, inverted_index: dict, tags_onehot: "pd.DataFrame"):
        super().__init__(inverted_index, tags_onehot)
        self.mineable_tags = set(VIBE_ADJACENCIES.keys()) | set(VIBE_CONTRASTS.keys())
        print(f"VibeTripletMiner initialized with {len(self.mineable_tags)} mineable vibe tags.")

    def mine(self, anchor_id: str, anchor_tags: set) -> tuple | None:
        
        # 1. Find a mineable vibe tag for the anchor
        anchor_vibe_tags = anchor_tags.intersection(self.mineable_tags)
        if not anchor_vibe_tags:
            return None
            
        anchor_vibe = random.choice(list(anchor_vibe_tags))
        
        # 2. Find Positive (Hard)
        positive_id = None
        adjacent_vibes = VIBE_ADJACENCIES.get(anchor_vibe)
        if adjacent_vibes:
            positive_vibe_tag = random.choice(list(adjacent_vibes))
            positive_pool = self._get_tracks_by_tag(positive_vibe_tag)
            positive_id = self._get_random_track_from_set(positive_pool, exclude_id=anchor_id)

        if not positive_id:
            # Fallback to an "Easy" positive (same tag)
            positive_pool = self._get_tracks_by_tag(anchor_vibe)
            positive_id = self._get_random_track_from_set(positive_pool, exclude_id=anchor_id)

        if not positive_id:
            return None # Cannot find a positive

        # 3. Find Negative (Hard)
        negative_id = None
        contrasting_vibes = VIBE_CONTRASTS.get(anchor_vibe)
        if contrasting_vibes:
            negative_vibe_tag = random.choice(list(contrasting_vibes))
            negative_pool = self._get_tracks_by_tag(negative_vibe_tag)
            negative_id = self._get_random_track_from_set(negative_pool, exclude_id=anchor_id)
            
        if not negative_id:
            # Fallback to a generic negative
            negative_pool = self._find_negative_by_exclusion(anchor_tags)
            negative_id = self._get_random_track_from_set(negative_pool, exclude_id=anchor_id)

        if not negative_id:
            return None # Cannot find a negative
            
        return positive_id, negative_id
