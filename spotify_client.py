import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyClient:
    def __init__(self):
        """
        Requires env vars:
        - SPOTIPY_CLIENT_ID
        - SPOTIPY_CLIENT_SECRET
        """
        auth_manager = SpotifyClientCredentials(
            client_id=os.environ.get("SPOTIPY_CLIENT_ID"),
            client_secret=os.environ.get("SPOTIPY_CLIENT_SECRET"),
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def enrich_tracks(self, track_ids):
        """
        Given a list of Spotify track IDs, return
        {track_id: {name, artist, album, spotify_url, preview_url, album_image}}
        """
        if not track_ids:
            return {}

        # Spotify limits to 50 IDs per call; handle if you ever have more
        meta = {}
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i : i + 50]
            resp = self.sp.tracks(batch)
            for t in resp["tracks"]:
                if t is None:
                    continue
                tid = t["id"]
                meta[tid] = {
                    "name": t["name"],
                    "artist": ", ".join(a["name"] for a in t["artists"]),
                    "album": t["album"]["name"],
                    "spotify_url": t["external_urls"]["spotify"],
                    "preview_url": t["preview_url"],
                    "album_image": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
                }
        return meta
