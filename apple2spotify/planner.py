from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from apple2spotify.apple_music import folder_path_for_playlist
from apple2spotify.models import AppleLibrary, ApplePlaylist, AppleTrack, SpotifyPlaylistSummary
from apple2spotify.utils import isoformat_or_none, normalize_text, similarity


SYSTEM_PLAYLIST_NAMES = {"Knihovna", "Hudba", "Staženo"}
_NUMBER_TOKEN_RE = re.compile(r"\d+")


def build_inventory(
    library: AppleLibrary,
    favorites_playlist_names: list[str],
    include_favorite_flags: bool = True,
) -> dict:
    favorite_track_ids = collect_liked_track_ids(
        library,
        favorites_playlist_names=favorites_playlist_names,
        include_favorite_flags=include_favorite_flags,
    )

    playlist_rows = []
    for playlist in library.playlists:
        playlist_rows.append(
            {
                "name": playlist.name,
                "playlist_id": playlist.playlist_id,
                "persistent_id": playlist.persistent_id,
                "folder_path": folder_path_for_playlist(library, playlist),
                "track_count": len(playlist.track_ids),
                "folder": playlist.folder,
                "visible": playlist.visible,
                "master": playlist.master,
                "distinguished_kind": playlist.distinguished_kind,
                "smart_playlist": playlist.is_smart,
                "is_user_playlist": is_user_playlist(playlist),
            }
        )

    return {
        "source_library": str(library.source_path),
        "exported_at": isoformat_or_none(library.exported_at),
        "track_count": len(library.tracks),
        "playlist_count": len(library.playlists),
        "favorites_playlist_names": favorites_playlist_names,
        "favorite_track_candidate_count": len(favorite_track_ids),
        "playlists": playlist_rows,
    }


def build_migration_plan(
    library: AppleLibrary,
    spotify_playlists: list[SpotifyPlaylistSummary],
    favorites_playlist_names: list[str],
    include_favorite_flags: bool = True,
) -> dict:
    spotify_lookup = _build_playlist_match_lookup(spotify_playlists)
    plan_playlists: list[dict] = []
    for playlist in library.playlists:
        if not is_user_playlist(playlist):
            continue
        if playlist.folder and not playlist.track_ids:
            continue

        folder_path = folder_path_for_playlist(library, playlist)
        proposed_match = _propose_playlist_match(playlist, spotify_lookup)
        notes: list[str] = []
        if playlist.is_smart:
            notes.append("Apple smart playlist rules cannot be recreated; current item snapshot will be used.")
        if playlist.folder:
            notes.append("Apple folder structure is used as context only; Spotify playlists are flat.")

        plan_playlists.append(
            {
                "apple_name": playlist.name,
                "apple_persistent_id": playlist.persistent_id,
                "apple_folder_path": folder_path,
                "apple_track_count": len(playlist.track_ids),
                "apple_is_smart": playlist.is_smart,
                "apple_is_folder": playlist.folder,
                "proposed_action": proposed_match["action"],
                "approval": "pending",
                "confidence": proposed_match["confidence"],
                "spotify_target": proposed_match["spotify_target"],
                "notes": notes,
            }
        )

    favorite_track_ids = collect_liked_track_ids(
        library,
        favorites_playlist_names=favorites_playlist_names,
        include_favorite_flags=include_favorite_flags,
    )

    return {
        "schema_version": 1,
        "source_library": str(library.source_path),
        "exported_at": isoformat_or_none(library.exported_at),
        "spotify_playlist_count": len(spotify_playlists),
        "settings": {
            "favorites_playlist_names": favorites_playlist_names,
            "include_favorite_flags": include_favorite_flags,
        },
        "favorites": {
            "sources": _describe_favorites_sources(favorites_playlist_names, include_favorite_flags),
            "candidate_track_count": len(favorite_track_ids),
            "proposed_action": "save",
            "approval": "pending",
        },
        "playlists": plan_playlists,
    }


def collect_liked_track_ids(
    library: AppleLibrary,
    favorites_playlist_names: list[str],
    include_favorite_flags: bool = True,
) -> list[int]:
    track_ids: set[int] = set()
    if include_favorite_flags:
        for track in library.tracks.values():
            if track.favorited or track.loved:
                track_ids.add(track.track_id)

    normalized_names = {normalize_text(name) for name in favorites_playlist_names}
    for playlist in library.playlists:
        if normalize_text(playlist.name) in normalized_names:
            track_ids.update(playlist.track_ids)

    return sorted(track_ids)


def is_user_playlist(playlist: ApplePlaylist) -> bool:
    if playlist.master:
        return False
    if playlist.distinguished_kind is not None:
        return False
    if playlist.name in SYSTEM_PLAYLIST_NAMES:
        return False
    return True


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apple_tracks_for_playlist(library: AppleLibrary, playlist_persistent_id: str | None) -> list[AppleTrack]:
    if not playlist_persistent_id:
        return []
    playlist = next(
        (item for item in library.playlists if item.persistent_id == playlist_persistent_id),
        None,
    )
    if not playlist:
        return []
    return [library.tracks[track_id] for track_id in playlist.track_ids if track_id in library.tracks]


def favorite_tracks(
    library: AppleLibrary,
    favorites_playlist_names: list[str],
    include_favorite_flags: bool = True,
) -> list[AppleTrack]:
    track_ids = collect_liked_track_ids(
        library,
        favorites_playlist_names=favorites_playlist_names,
        include_favorite_flags=include_favorite_flags,
    )
    return [library.tracks[track_id] for track_id in track_ids if track_id in library.tracks]


def _build_playlist_match_lookup(spotify_playlists: list[SpotifyPlaylistSummary]) -> dict[str, list[SpotifyPlaylistSummary]]:
    lookup: dict[str, list[SpotifyPlaylistSummary]] = defaultdict(list)
    for playlist in spotify_playlists:
        lookup[normalize_text(playlist.name)].append(playlist)
    return lookup


def _propose_playlist_match(
    apple_playlist: ApplePlaylist,
    spotify_lookup: dict[str, list[SpotifyPlaylistSummary]],
) -> dict:
    normalized_name = normalize_text(apple_playlist.name)
    exact_matches = spotify_lookup.get(normalized_name, [])
    if exact_matches:
        chosen = sorted(exact_matches, key=lambda item: item.track_total, reverse=True)[0]
        return {
            "action": "update",
            "confidence": 1.0,
            "spotify_target": _serialize_spotify_playlist(chosen),
        }

    best_match = None
    best_score = 0.0
    for candidates in spotify_lookup.values():
        for candidate in candidates:
            if not _playlist_number_tokens_compatible(apple_playlist.name, candidate.name):
                continue
            score = similarity(apple_playlist.name, candidate.name)
            if score > best_score:
                best_score = score
                best_match = candidate

    if best_match and best_score >= 0.88:
        return {
            "action": "update",
            "confidence": round(best_score, 4),
            "spotify_target": _serialize_spotify_playlist(best_match),
        }

    return {
        "action": "create",
        "confidence": round(best_score, 4),
        "spotify_target": None,
    }


def _serialize_spotify_playlist(playlist: SpotifyPlaylistSummary) -> dict:
    return {
        "id": playlist.playlist_id,
        "name": playlist.name,
        "owner_id": playlist.owner_id,
        "public": playlist.public,
        "snapshot_id": playlist.snapshot_id,
        "track_total": playlist.track_total,
    }


def _describe_favorites_sources(favorites_playlist_names: list[str], include_favorite_flags: bool) -> list[str]:
    sources = [f"playlist:{name}" for name in favorites_playlist_names]
    if include_favorite_flags:
        sources.extend(["apple_flag:Favorited", "apple_flag:Loved"])
    return sources


def _playlist_number_tokens_compatible(left: str, right: str) -> bool:
    left_tokens = set(_NUMBER_TOKEN_RE.findall(left))
    right_tokens = set(_NUMBER_TOKEN_RE.findall(right))
    if not left_tokens or not right_tokens:
        return True
    return left_tokens == right_tokens