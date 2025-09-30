# main.py

import sys
from spotify_agent import IntelligentShuffler
from dotenv import load_dotenv

load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <spotify_playlist_url_or_id>")
        sys.exit(1)

    playlist_input = sys.argv[1]
    
    # Extract playlist ID from URL or use it directly
    if "playlist/" in playlist_input:
        playlist_id = playlist_input.split("playlist/")[1].split("?")[0]
    else:
        playlist_id = playlist_input

    print(f"Targeting playlist ID: {playlist_id}")

    agent = IntelligentShuffler(playlist_id)
    
    try:
        agent.initialize()
        agent.run()
    except KeyboardInterrupt:
        print("\nShutting down agent. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your Spotify credentials and network connection.")

if __name__ == "__main__":
    main()