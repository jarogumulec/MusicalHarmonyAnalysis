from __future__ import annotations

import plistlib
from datetime import datetime
from pathlib import Path

from apple2spotify.models import AppleLibrary, ApplePlaylist, AppleTrack


def load_apple_library(path: str | Path) -> AppleLibrary:
    source_path = Path(path)
    with source_path.open("rb") as handle:
        raw = plistlib.load(handle)

    raw_tracks = raw.get("Tracks", {})
    tracks: dict[int, AppleTrack] = {}
    for track_id_text, track_data in raw_tracks.items():
        track_id = int(track_id_text)
        tracks[track_id] = AppleTrack(
            track_id=track_id,
            persistent_id=track_data.get("Persistent ID"),
            name=track_data.get("Name") or f"Track {track_id}",
            artist=track_data.get("Artist"),
            album=track_data.get("Album"),
            album_artist=track_data.get("Album Artist"),
            composer=track_data.get("Composer"),
            total_time_ms=track_data.get("Total Time"),
            year=track_data.get("Year"),
            date_added=_datetime_or_none(track_data.get("Date Added")),
            date_modified=_datetime_or_none(track_data.get("Date Modified")),
            track_type=track_data.get("Track Type"),
            location=track_data.get("Location"),
            favorited=bool(track_data.get("Favorited")),
            loved=bool(track_data.get("Loved")),
        )

    playlists: list[ApplePlaylist] = []
    for playlist_data in raw.get("Playlists", []):
        items = playlist_data.get("Playlist Items", [])
        playlists.append(
            ApplePlaylist(
                name=playlist_data.get("Name") or "Unnamed Playlist",
                playlist_id=playlist_data.get("Playlist ID"),
                persistent_id=playlist_data.get("Playlist Persistent ID"),
                parent_persistent_id=playlist_data.get("Parent Persistent ID"),
                folder=bool(playlist_data.get("Folder")),
                visible=playlist_data.get("Visible"),
                master=bool(playlist_data.get("Master")),
                distinguished_kind=playlist_data.get("Distinguished Kind"),
                smart_info_present="Smart Info" in playlist_data,
                smart_criteria_present="Smart Criteria" in playlist_data,
                track_ids=[int(item["Track ID"]) for item in items if "Track ID" in item],
            )
        )

    return AppleLibrary(
        source_path=source_path,
        exported_at=_datetime_or_none(raw.get("Date")),
        tracks=tracks,
        playlists=playlists,
        library_persistent_id=raw.get("Library Persistent ID"),
        music_folder=raw.get("Music Folder"),
    )


def folder_path_for_playlist(library: AppleLibrary, playlist: ApplePlaylist) -> str:
    lookup = library.playlist_lookup()
    parts: list[str] = []
    current = playlist
    seen: set[str] = set()
    while current.parent_persistent_id:
        parent_id = current.parent_persistent_id
        if parent_id in seen:
            break
        seen.add(parent_id)
        parent = lookup.get(parent_id)
        if not parent:
            break
        parts.append(parent.name)
        current = parent
    return " / ".join(reversed(parts))


def _datetime_or_none(value: object) -> datetime | None:
    return value if isinstance(value, datetime) else None