#!/usr/bin/env python3
"""
Create Playlist Script - Entry Point

This script creates a playlist from all liked songs.
"""

import sys
import os

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.create_playlist import create_playlist_from_all_liked

if __name__ == "__main__":
    create_playlist_from_all_liked()
