# create_playlist.py

import spotipy
from tqdm import tqdm
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

def create_playlist_from_all_liked():
    """
    Creates a new Spotify playlist populated with ALL of the user's
    "Liked Songs".
    """
    load_dotenv()
    
    scope = "playlist-modify-public user-library-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    
    user_id = sp.me()['id']
    
    playlist_name = input("Enter a name for your new AI playlist (e.g., 'Full Liked Songs AI'): ")
    if not playlist_name:
        playlist_name = "My Full AI Playlist"
        
    playlist_desc = "A dynamic playlist created from all my Liked Songs, managed by a personal AI agent."
    
    print("\nFetching ALL of your Liked Songs. This may take a while for large libraries...")
    
    # --- Mentorship Note: Handling Pagination ---
    # The Spotify API returns a 'paginated' object. We get the first page,
    # then we loop using `sp.next()` as long as there is a 'next' page URL.
    # This ensures we retrieve every single song, not just the first 50.
    
    liked_songs_results = sp.current_user_saved_tracks(limit=50)
    all_tracks = liked_songs_results['items']
    
    # Use tqdm for a nice progress bar
    with tqdm(total=liked_songs_results['total'], desc="Fetching liked songs") as pbar:
        pbar.update(len(liked_songs_results['items']))
        while liked_songs_results['next']:
            liked_songs_results = sp.next(liked_songs_results)
            all_tracks.extend(liked_songs_results['items'])
            pbar.update(len(liked_songs_results['items']))

    if not all_tracks:
        print("Could not find any songs in your 'Liked Songs'. Please add some and try again.")
        return
        
    track_uris = [item['track']['uri'] for item in all_tracks]
    print(f"\nFound a total of {len(track_uris)} liked songs.")

    try:
        playlist = sp.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=True,
            description=playlist_desc
        )
        playlist_id = playlist['id']
        playlist_url = playlist['external_urls']['spotify']
        print(f"Successfully created playlist '{playlist_name}'.")
    except Exception as e:
        print(f"Error creating playlist: {e}")
        return
        
    # --- Mentorship Note: Batching Requests ---
    # The API for adding items to a playlist has a limit of 100 tracks per
    # request. We loop through our list of track URIs in chunks of 100
    # to add them all without hitting the API limit.
    print("Adding songs to the new playlist in batches...")
    for i in tqdm(range(0, len(track_uris), 100), desc="Adding songs"):
        batch = track_uris[i:i+100]
        try:
            sp.playlist_add_items(playlist_id, batch)
        except Exception as e:
            print(f"\nError adding a batch of songs: {e}")
            
    print("\n--- Your Playlist is Ready! ---")
    print(f"URL: {playlist_url}")
    print("---------------------------------")
    print("\nYou can now use this URL to run the main agent.")

if __name__ == "__main__":
    create_playlist_from_all_liked()