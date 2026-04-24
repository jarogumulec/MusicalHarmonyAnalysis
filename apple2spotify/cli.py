from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from apple2spotify.apple_music import load_apple_library
from apple2spotify.planner import (
    apple_tracks_for_playlist,
    build_inventory,
    build_migration_plan,
    favorite_tracks,
    load_json,
    save_json,
)
from apple2spotify.spotify_client import (
    SpotifySyncClient,
    load_track_match_cache,
    save_track_match_cache,
)
from apple2spotify.utils import load_dotenv


DEFAULT_LIBRARY_XML = Path("apple2spotify/apple_music_Knihovna.xml")
DEFAULT_PLAN_PATH = Path("apple2spotify/migration_plan.json")
DEFAULT_INVENTORY_PATH = Path("apple2spotify/apple_inventory.json")
DEFAULT_CACHE_PATH = Path("apple2spotify/spotify_track_cache.json")
DEFAULT_REPORT_PATH = Path("apple2spotify/sync_report.json")
DEFAULT_SPOTIFY_AUTH_CACHE = Path("apple2spotify/.spotify_auth_cache")
DEFAULT_SPOTIFY_ENV_PATH = Path("apple2spotify/.env")
DEFAULT_FAVORITES_PLAYLIST = "Oblíbené_manual_4_2026"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apple Music → Spotify migration helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-apple", help="Parse the Apple library export and write inventory JSON")
    _add_shared_library_args(inspect_parser)
    inspect_parser.add_argument("--output", type=Path, default=DEFAULT_INVENTORY_PATH)
    inspect_parser.set_defaults(func=inspect_apple_command)

    plan_parser = subparsers.add_parser("build-plan", help="Read Apple export and Spotify state and write approval plan JSON")
    _add_shared_library_args(plan_parser)
    plan_parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH)
    plan_parser.add_argument("--spotify-auth-cache", type=Path, default=DEFAULT_SPOTIFY_AUTH_CACHE)
    plan_parser.set_defaults(func=build_plan_command)

    apply_parser = subparsers.add_parser("apply-plan", help="Apply approved actions from a plan file")
    _add_shared_library_args(apply_parser)
    apply_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    apply_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    apply_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    apply_parser.add_argument("--spotify-auth-cache", type=Path, default=DEFAULT_SPOTIFY_AUTH_CACHE)
    apply_parser.add_argument("--public-playlists", action="store_true")
    apply_parser.add_argument("--only-favorites", action="store_true")
    apply_parser.add_argument("--favorites-offset", type=int, default=0)
    apply_parser.add_argument("--favorites-limit", type=int)
    apply_parser.add_argument(
        "--favorites-added-after",
        help="Only process Apple favorites added on or after YYYY-MM-DD.",
    )
    apply_parser.add_argument(
        "--favorites-newest-first",
        action="store_true",
        help="Process Apple favorites from newest to oldest by Date Added.",
    )
    apply_parser.add_argument(
        "--merge-added-after",
        type=str,
        help="Only process Apple tracks added on or after YYYY-MM-DD. Skips older tracks.",
    )
    apply_parser.add_argument(
        "--stop-after-n-existing",
        type=int,
        help="Stop processing a playlist after N consecutive tracks already exist in Spotify.",
    )
    apply_parser.add_argument(
        "--search-delay",
        type=float,
        default=0.3,
        help="Seconds to wait between Spotify search requests (default: 0.3). Increase to avoid rate limits.",
    )
    apply_parser.set_defaults(func=apply_plan_command)

    backup_parser = subparsers.add_parser("backup-spotify", help="Export Spotify playlists and liked tracks to JSON files")
    backup_parser.add_argument("--output-dir", type=Path)
    backup_parser.add_argument("--spotify-auth-cache", type=Path, default=DEFAULT_SPOTIFY_AUTH_CACHE)
    backup_parser.add_argument("--spotify-env-file", type=Path, default=DEFAULT_SPOTIFY_ENV_PATH)
    backup_parser.set_defaults(func=backup_spotify_command)

    dedupe_parser = subparsers.add_parser("dedupe-playlist", help="Remove duplicate track occurrences from a Spotify playlist")
    dedupe_parser.add_argument("--playlist-id", required=True)
    dedupe_parser.add_argument("--spotify-auth-cache", type=Path, default=DEFAULT_SPOTIFY_AUTH_CACHE)
    dedupe_parser.add_argument("--spotify-env-file", type=Path, default=DEFAULT_SPOTIFY_ENV_PATH)
    dedupe_parser.set_defaults(func=dedupe_playlist_command)

    return parser


def inspect_apple_command(args: argparse.Namespace) -> int:
    library = load_apple_library(args.library_xml)
    inventory = build_inventory(
        library,
        favorites_playlist_names=args.favorites_playlist,
        include_favorite_flags=not args.no_favorite_flags,
    )
    _ensure_parent(args.output)
    save_json(args.output, inventory)
    print(f"Apple inventory written to {args.output}")
    print(f"Tracks: {inventory['track_count']}; playlists: {inventory['playlist_count']}; favorite candidates: {inventory['favorite_track_candidate_count']}")
    return 0


def build_plan_command(args: argparse.Namespace) -> int:
    library = load_apple_library(args.library_xml)
    _ensure_spotify_environment(args.spotify_env_file)
    spotify = SpotifySyncClient(cache_path=args.spotify_auth_cache)
    spotify_playlists = spotify.list_playlists()
    plan = build_migration_plan(
        library,
        spotify_playlists=spotify_playlists,
        favorites_playlist_names=args.favorites_playlist,
        include_favorite_flags=not args.no_favorite_flags,
    )
    _ensure_parent(args.output)
    save_json(args.output, plan)
    print(f"Migration plan written to {args.output}")
    print("Edit approval fields before running apply-plan.")
    return 0


def apply_plan_command(args: argparse.Namespace) -> int:
    library = load_apple_library(args.library_xml)
    plan = load_json(args.plan)
    _ensure_spotify_environment(args.spotify_env_file)
    spotify = SpotifySyncClient(cache_path=args.spotify_auth_cache)
    cache = load_track_match_cache(args.cache)
    settings = plan.get("settings", {})
    favorites_playlist_names = settings.get("favorites_playlist_names") or args.favorites_playlist
    include_favorite_flags = settings.get("include_favorite_flags")
    if include_favorite_flags is None:
        include_favorite_flags = not args.no_favorite_flags

    merge_added_after = _parse_iso_date(args.merge_added_after) if args.merge_added_after else None
    stop_after_n_existing = args.stop_after_n_existing
    search_delay = args.search_delay

    report = {
        "plan": str(args.plan),
        "playlist_results": [],
        "favorites_result": None,
    }

    for playlist_plan in plan.get("playlists", []):
        if args.only_favorites:
            break
        approval = playlist_plan.get("approval")
        if approval not in {"approved_create", "approved_update"}:
            continue

        try:
            target_info = playlist_plan.get("spotify_target") or {}
            playlist_name = target_info.get("name") or playlist_plan["apple_name"]

            # Determine target_id and existing_ids first
            target_id = None
            existing_ids = set()
            if approval == "approved_create":
                existing_target_id = target_info.get("id")
                if existing_target_id:
                    target_id = existing_target_id
                    existing_ids = spotify.get_playlist_track_ids(target_id)
            else:
                target_id = target_info.get("id")
                if not target_id:
                    raise RuntimeError(
                        f"Playlist '{playlist_plan['apple_name']}' is approved for update but has no spotify target id."
                    )
                existing_ids = spotify.get_playlist_track_ids(target_id)

            tracks = apple_tracks_for_playlist(library, playlist_plan.get("apple_persistent_id"))
            if merge_added_after is not None:
                tracks = [
                    track for track in tracks
                    if track.date_added is not None and track.date_added.date() >= merge_added_after
                ]
            resolved_ids, unresolved = _resolve_tracks(
                spotify, cache, tracks,
                existing_playlist_ids=existing_ids,
                stop_after_n_existing=stop_after_n_existing,
            )
            if missing_ids:
                spotify.add_tracks_to_playlist(target_id, missing_ids)

            stopped_early = any(r.get("reason", "").startswith("stopped_after_") for r in unresolved)
            report["playlist_results"].append(
                {
                    "apple_name": playlist_plan["apple_name"],
                    "approval": approval,
                    "status": "success" if not stopped_early else "success_stopped_early",
                    "spotify_target_id": target_id,
                    "resolved_track_count": len(resolved_ids),
                    "added_track_count": len(missing_ids),
                    "unresolved_tracks": unresolved,
                    "stopped_early": stopped_early,
                }
            )
            if approval == "approved_create" and playlist_plan.get("spotify_target", {}).get("id"):
                playlist_plan["approval"] = "approved_update"
                playlist_plan["proposed_action"] = "update"
        except Exception as exc:
            report["playlist_results"].append(
                {
                    "apple_name": playlist_plan["apple_name"],
                    "approval": approval,
                    "status": "error",
                    "error": str(exc),
                }
            )

    favorites_plan = plan.get("favorites", {})
    if favorites_plan.get("approval") == "approved_save":
        try:
            tracks = favorite_tracks(
                library,
                favorites_playlist_names=favorites_playlist_names,
                include_favorite_flags=include_favorite_flags,
            )
            # Apply both --merge-added-after (global) and --favorites-added-after (favorites-specific)
            cutoff = merge_added_after
            added_after = _parse_iso_date(args.favorites_added_after) if args.favorites_added_after else None
            if added_after is not None and (cutoff is None or added_after > cutoff):
                cutoff = added_after
            if cutoff is not None:
                tracks = [
                    track
                    for track in tracks
                    if track.date_added is not None and track.date_added.date() >= cutoff
                ]
            if args.favorites_newest_first:
                tracks = sorted(
                    tracks,
                    key=lambda track: (track.date_added is not None, track.date_added),
                    reverse=True,
                )
            total_candidate_count = len(tracks)
            start = max(args.favorites_offset, 0)
            end = start + args.favorites_limit if args.favorites_limit is not None else total_candidate_count
            tracks = tracks[start:end]
            resolved_ids, unresolved = _resolve_tracks(
                spotify, cache, tracks,
                existing_playlist_ids=None,  # For favorites, we check against saved tracks later
                stop_after_n_existing=None,  # Not applicable for favorites
            )
            saved_ids = spotify.get_saved_track_ids()
            missing_ids = [track_id for track_id in resolved_ids if track_id not in saved_ids]
            if missing_ids:
                spotify.save_tracks(missing_ids)
            report["favorites_result"] = {
                "status": "success",
                "batch_offset": start,
                "batch_limit": args.favorites_limit,
                "batch_track_count": len(tracks),
                "total_candidate_count": total_candidate_count,
                "favorites_added_after": args.favorites_added_after,
                "favorites_newest_first": args.favorites_newest_first,
                "resolved_track_count": len(resolved_ids),
                "added_track_count": len(missing_ids),
                "unresolved_tracks": unresolved,
            }
        except Exception as exc:
            report["favorites_result"] = {
                "status": "error",
                "error": str(exc),
            }

    _ensure_parent(args.cache)
    save_track_match_cache(args.cache, cache)
    _ensure_parent(args.report)
    save_json(args.report, report)
    save_json(args.plan, plan)
    print(f"Sync report written to {args.report}")
    print(f"Track match cache written to {args.cache}")
    return 0


def backup_spotify_command(args: argparse.Namespace) -> int:
    _ensure_spotify_environment(args.spotify_env_file)
    spotify = SpotifySyncClient(cache_path=args.spotify_auth_cache)
    output_dir = args.output_dir or Path("apple2spotify/spotify_backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    playlists_dir = output_dir / "playlist_items"
    playlists_dir.mkdir(parents=True, exist_ok=True)

    profile = spotify.current_user
    playlists = spotify.list_playlists()
    playlist_rows = []
    backup_errors = []
    for playlist in playlists:
        backup_file = None
        try:
            items = spotify.get_playlist_items(playlist.playlist_id)
            serialized_items = [_serialize_playlist_item(item) for item in items]
            backup_file = f"playlist_items/{playlist.playlist_id}.json"
            (playlists_dir / f"{playlist.playlist_id}.json").write_text(
                json.dumps(serialized_items, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            backup_errors.append(
                {
                    "playlist_id": playlist.playlist_id,
                    "playlist_name": playlist.name,
                    "error": str(exc),
                }
            )
        playlist_rows.append(
            {
                "id": playlist.playlist_id,
                "name": playlist.name,
                "owner_id": playlist.owner_id,
                "public": playlist.public,
                "snapshot_id": playlist.snapshot_id,
                "track_total": playlist.track_total,
                "backup_items_file": backup_file,
            }
        )

    liked_tracks = [_serialize_saved_track(item) for item in spotify.get_saved_tracks()]

    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "playlists.json").write_text(json.dumps(playlist_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "liked_tracks.json").write_text(json.dumps(liked_tracks, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "backup_errors.json").write_text(json.dumps(backup_errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Spotify backup written to {output_dir}")
    print(f"Playlists: {len(playlist_rows)}; liked tracks: {len(liked_tracks)}; skipped playlists: {len(backup_errors)}")
    return 0


def dedupe_playlist_command(args: argparse.Namespace) -> int:
    _ensure_spotify_environment(args.spotify_env_file)
    spotify = SpotifySyncClient(cache_path=args.spotify_auth_cache)
    items = spotify.get_playlist_items(args.playlist_id)
    positions_by_track: dict[str, list[int]] = defaultdict(list)
    uri_by_track: dict[str, str] = {}

    for index, item in enumerate(items):
        track = item.get("item")
        if not track or track.get("type") != "track" or not track.get("id"):
            continue
        track_id = track["id"]
        positions_by_track[track_id].append(index)
        uri_by_track[track_id] = track.get("uri") or f"spotify:track:{track_id}"

    removals: list[dict] = []
    duplicate_occurrences = 0
    for track_id, positions in positions_by_track.items():
        if len(positions) <= 1:
            continue
        duplicate_occurrences += len(positions) - 1
        removals.append({"uri": uri_by_track[track_id], "positions": positions[1:]})

    if not removals:
        print("No duplicates found.")
        return 0

    spotify.remove_playlist_items(args.playlist_id, removals)
    print(f"Removed {duplicate_occurrences} duplicate playlist item(s) from {args.playlist_id}")
    return 0


def _resolve_tracks(
    spotify: SpotifySyncClient,
    cache: dict[str, dict],
    tracks: list,
    existing_playlist_ids: set[str] | None = None,
    stop_after_n_existing: int | None = None,
) -> tuple[list[str], list[dict]]:
    resolved_ids: list[str] = []
    unresolved: list[dict] = []
    consecutive_existing = 0

    for track in tracks:
        cached = cache.get(track.cache_key)
        if cached:
            if cached.get("spotify_track_id"):
                resolved_ids.append(cached["spotify_track_id"])
                # Check if this track already exists in the playlist
                if existing_playlist_ids and cached["spotify_track_id"] in existing_playlist_ids:
                    consecutive_existing += 1
                    if stop_after_n_existing and consecutive_existing >= stop_after_n_existing:
                        unresolved.append({
                            "track": track.display_name,
                            "reason": f"stopped_after_{consecutive_existing}_existing_tracks",
                        })
                        break
                else:
                    consecutive_existing = 0
            else:
                unresolved.append({"track": track.display_name, **cached})
                consecutive_existing = 0
            continue

        try:
            match = spotify.search_best_track_match(track)
        except Exception as search_err:
            error_msg = str(search_err)
            payload = {
                "spotify_track_id": None,
                "spotify_name": None,
                "spotify_artist": None,
                "spotify_album": None,
                "confidence": 0.0,
                "reason": "search_error",
                "raw_query": None,
                "error": error_msg,
            }
            cache[track.cache_key] = payload
            unresolved.append({"track": track.display_name, **payload})
            consecutive_existing = 0
            continue

        time.sleep(search_delay)

        payload = {
            "spotify_track_id": match.spotify_track_id,
            "spotify_name": match.spotify_name,
            "spotify_artist": match.spotify_artist,
            "spotify_album": match.spotify_album,
            "confidence": match.confidence,
            "reason": match.reason,
            "raw_query": match.raw_query,
        }
        cache[track.cache_key] = payload
        if match.spotify_track_id:
            resolved_ids.append(match.spotify_track_id)
            # Check if this track already exists in the playlist
            if existing_playlist_ids and match.spotify_track_id in existing_playlist_ids:
                consecutive_existing += 1
                if stop_after_n_existing and consecutive_existing >= stop_after_n_existing:
                    unresolved.append({
                        "track": track.display_name,
                        "reason": f"stopped_after_{consecutive_existing}_existing_tracks",
                    })
                    break
            else:
                consecutive_existing = 0
        else:
            unresolved.append({"track": track.display_name, **payload})
            consecutive_existing = 0

    deduped_ids = list(dict.fromkeys(resolved_ids))
    return deduped_ids, unresolved


def _add_shared_library_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--library-xml", type=Path, default=DEFAULT_LIBRARY_XML)
    parser.add_argument(
        "--favorites-playlist",
        action="append",
        default=[DEFAULT_FAVORITES_PLAYLIST],
        help="Playlist names to union into Spotify Liked Songs candidate set.",
    )
    parser.add_argument(
        "--no-favorite-flags",
        action="store_true",
        help="Ignore Apple Favorited/Loved flags and use only playlist-based favorites.",
    )
    parser.add_argument(
        "--spotify-env-file",
        type=Path,
        default=DEFAULT_SPOTIFY_ENV_PATH,
        help="Path to a .env file with SPOTIPY_* credentials.",
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_spotify_environment(env_file: Path) -> None:
    load_dotenv(env_file)
    missing = [
        key
        for key in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI")
        if not os.environ.get(key)
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing Spotify credentials: "
            f"{joined}. Set them in the shell or in {env_file}."
        )


def _parse_iso_date(value: str):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def _serialize_playlist_item(item: dict) -> dict:
    track = item.get("item") or {}
    artists = [artist.get("name") for artist in track.get("artists", []) if artist.get("name")]
    album = track.get("album") or {}
    return {
        "added_at": item.get("added_at"),
        "added_by": (item.get("added_by") or {}).get("id"),
        "is_local": item.get("is_local"),
        "type": track.get("type"),
        "id": track.get("id"),
        "uri": track.get("uri"),
        "name": track.get("name"),
        "artists": artists,
        "album": album.get("name"),
        "duration_ms": track.get("duration_ms"),
    }


def _serialize_saved_track(item: dict) -> dict:
    track = item.get("track") or {}
    artists = [artist.get("name") for artist in track.get("artists", []) if artist.get("name")]
    album = track.get("album") or {}
    return {
        "added_at": item.get("added_at"),
        "id": track.get("id"),
        "uri": track.get("uri"),
        "name": track.get("name"),
        "artists": artists,
        "album": album.get("name"),
        "duration_ms": track.get("duration_ms"),
    }


if __name__ == "__main__":
    sys.exit(main())