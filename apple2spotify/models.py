from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class AppleTrack:
    track_id: int
    persistent_id: str | None
    name: str
    artist: str | None
    album: str | None
    album_artist: str | None
    composer: str | None
    total_time_ms: int | None
    year: int | None
    date_added: datetime | None
    date_modified: datetime | None
    track_type: str | None
    location: str | None
    favorited: bool = False
    loved: bool = False

    @property
    def display_name(self) -> str:
        artist = self.artist or self.album_artist or "Unknown Artist"
        return f"{artist} — {self.name}"

    @property
    def cache_key(self) -> str:
        persistent = self.persistent_id or "no-persistent-id"
        duration = self.total_time_ms or 0
        return f"{persistent}:{self.name}:{self.artist}:{self.album}:{duration}"


@dataclass(slots=True)
class ApplePlaylist:
    name: str
    playlist_id: int | None
    persistent_id: str | None
    parent_persistent_id: str | None
    folder: bool
    visible: bool | None
    master: bool
    distinguished_kind: int | None
    smart_info_present: bool
    smart_criteria_present: bool
    track_ids: list[int] = field(default_factory=list)

    @property
    def has_tracks(self) -> bool:
        return bool(self.track_ids)

    @property
    def is_smart(self) -> bool:
        return self.smart_info_present or self.smart_criteria_present


@dataclass(slots=True)
class AppleLibrary:
    source_path: Path
    exported_at: datetime | None
    tracks: dict[int, AppleTrack]
    playlists: list[ApplePlaylist]
    library_persistent_id: str | None
    music_folder: str | None

    def playlist_lookup(self) -> dict[str, ApplePlaylist]:
        return {
            playlist.persistent_id: playlist
            for playlist in self.playlists
            if playlist.persistent_id
        }


@dataclass(slots=True)
class SpotifyPlaylistSummary:
    playlist_id: str
    name: str
    owner_id: str | None
    public: bool | None
    snapshot_id: str | None
    track_total: int


@dataclass(slots=True)
class TrackMatchResult:
    spotify_track_id: str | None
    spotify_name: str | None
    spotify_artist: str | None
    spotify_album: str | None
    confidence: float
    reason: str
    raw_query: str