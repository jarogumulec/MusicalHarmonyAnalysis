from __future__ import annotations

import json
from pathlib import Path

import requests
import spotipy
from spotipy import SpotifyException
from spotipy.oauth2 import SpotifyOAuth

from apple2spotify.models import AppleTrack, SpotifyPlaylistSummary, TrackMatchResult
from apple2spotify.utils import chunked, normalize_text, normalize_title, similarity


DEFAULT_SCOPES = [
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-library-read",
    "user-library-modify",
]


class SpotifySyncClient:
    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path
        auth_manager = SpotifyOAuth(
            scope=" ".join(DEFAULT_SCOPES),
            open_browser=True,
            cache_path=str(cache_path) if cache_path else None,
        )
        self.client = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=30, retries=0)
        self.current_user = self.client.current_user()

    def list_playlists(self) -> list[SpotifyPlaylistSummary]:
        playlists: list[SpotifyPlaylistSummary] = []
        response = self.client.current_user_playlists(limit=50)
        while response:
            for item in response.get("items", []):
                playlists.append(
                    SpotifyPlaylistSummary(
                        playlist_id=item["id"],
                        name=item["name"],
                        owner_id=(item.get("owner") or {}).get("id"),
                        public=item.get("public"),
                        snapshot_id=item.get("snapshot_id"),
                        track_total=(item.get("tracks") or {}).get("total", 0),
                    )
                )
            response = self.client.next(response) if response.get("next") else None
        return playlists

    def get_playlist_track_ids(self, playlist_id: str) -> set[str]:
        track_ids: set[str] = set()
        for item in self.get_playlist_items(playlist_id):
            track = item.get("item")
            if track and track.get("type") == "track" and track.get("id"):
                track_ids.add(track["id"])
        return track_ids

    def get_playlist_items(self, playlist_id: str) -> list[dict]:
        items: list[dict] = []
        response = self._api_request(
            "GET",
            f"playlists/{playlist_id}/items",
            params={"limit": 100},
        )
        while response:
            items.extend(response.get("items", []))
            next_url = response.get("next")
            response = self._api_request("GET", next_url) if next_url else None
        return items

    def get_saved_track_ids(self) -> set[str]:
        saved_ids: set[str] = set()
        response = self.client.current_user_saved_tracks(limit=50)
        while response:
            for item in response.get("items", []):
                track = item.get("track")
                if track and track.get("id"):
                    saved_ids.add(track["id"])
            response = self.client.next(response) if response.get("next") else None
        return saved_ids

    def create_playlist(self, name: str, public: bool = False, description: str | None = None) -> SpotifyPlaylistSummary:
        playlist = self.client._post(
            "me/playlists",
            payload={
                "name": name,
                "public": public,
                "description": description,
            },
        )
        return SpotifyPlaylistSummary(
            playlist_id=playlist["id"],
            name=playlist["name"],
            owner_id=(playlist.get("owner") or {}).get("id"),
            public=playlist.get("public"),
            snapshot_id=playlist.get("snapshot_id"),
            track_total=(playlist.get("tracks") or {}).get("total", 0),
        )

    def add_tracks_to_playlist(self, playlist_id: str, spotify_track_ids: list[str]) -> None:
        for batch in chunked(spotify_track_ids, 100):
            self._api_request(
                "POST",
                f"playlists/{playlist_id}/items",
                json_body={"uris": [f"spotify:track:{track_id}" for track_id in batch]},
            )

    def remove_playlist_items(self, playlist_id: str, tracks: list[dict]) -> None:
        for batch in _chunk_dicts(tracks, 100):
            self._api_request(
                "DELETE",
                f"playlists/{playlist_id}/items",
                json_body={"items": batch},
            )

    def save_tracks(self, spotify_track_ids: list[str]) -> None:
        uris = [f"spotify:track:{track_id}" for track_id in spotify_track_ids]
        for batch in chunked(uris, 40):
            self._api_request(
                "PUT",
                "me/library",
                params={"uris": ",".join(batch)},
            )

    def get_saved_tracks(self) -> list[dict]:
        items: list[dict] = []
        response = self.client.current_user_saved_tracks(limit=50)
        while response:
            items.extend(response.get("items", []))
            response = self.client.next(response) if response.get("next") else None
        return items

    def search_best_track_match(self, apple_track: AppleTrack) -> TrackMatchResult:
        query = _build_track_query(apple_track)
        candidates = self._spotify_search(query)
        items = ((candidates.get("tracks") or {}).get("items")) or []
        if not items:
            fallback_query = f"{apple_track.name} {apple_track.artist or apple_track.album_artist or ''}".strip()
            candidates = self._spotify_search(fallback_query)
            items = ((candidates.get("tracks") or {}).get("items")) or []
            query = fallback_query

        best_item = None
        best_score = -1.0
        second_best = -1.0
        for item in items:
            score = _score_track_candidate(apple_track, item)
            if score > best_score:
                second_best = best_score
                best_item = item
                best_score = score
            elif score > second_best:
                second_best = score

        if not best_item:
            return TrackMatchResult(
                spotify_track_id=None,
                spotify_name=None,
                spotify_artist=None,
                spotify_album=None,
                confidence=0.0,
                reason="no_results",
                raw_query=query,
            )

        if best_score < 0.78:
            reason = "low_confidence"
        elif second_best >= 0.0 and best_score - second_best < 0.08:
            reason = "ambiguous"
        else:
            reason = "matched"

        artists = ", ".join(artist["name"] for artist in best_item.get("artists", []))
        return TrackMatchResult(
            spotify_track_id=best_item.get("id") if reason == "matched" else None,
            spotify_name=best_item.get("name"),
            spotify_artist=artists or None,
            spotify_album=(best_item.get("album") or {}).get("name"),
            confidence=round(best_score, 4),
            reason=reason,
            raw_query=query,
        )

    def _spotify_search(self, query: str) -> dict:
        try:
            return self.client.search(q=query, type="track", limit=10)
        except SpotifyException as exc:
            if exc.http_status == 429:
                raise RuntimeError(
                    f"Spotify search rate-limited. Retry later. Details: {exc.msg}"
                ) from exc
            raise RuntimeError(f"Spotify search failed: {exc.msg}") from exc

    def _api_request(
        self,
        method: str,
        path_or_url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> dict:
        if path_or_url.startswith("https://"):
            url = path_or_url
        else:
            url = f"https://api.spotify.com/v1/{path_or_url.lstrip('/')}"

        token = self.client.auth_manager.get_access_token(as_dict=False)
        if isinstance(token, dict):
            access_token = token.get("access_token")
        else:
            access_token = token

        response = requests.request(
            method=method,
            url=url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            json=json_body,
            timeout=30,
        )
        if response.status_code >= 400:
            message = _extract_error_message(response)
            raise RuntimeError(f"Spotify API {method} {url} failed: {response.status_code} {message}")
        if not response.text:
            return {}
        return response.json()


def _build_track_query(track: AppleTrack) -> str:
    artist = track.artist or track.album_artist or ""
    query_parts = [f'track:"{track.name}"']
    if artist:
        query_parts.append(f'artist:"{artist}"')
    if track.album:
        query_parts.append(f'album:"{track.album}"')
    return " ".join(query_parts)


def _score_track_candidate(apple_track: AppleTrack, spotify_item: dict) -> float:
    spotify_name = spotify_item.get("name") or ""
    spotify_artists = ", ".join(artist.get("name", "") for artist in spotify_item.get("artists", []))
    spotify_album = (spotify_item.get("album") or {}).get("name") or ""
    name_score = max(
        similarity(apple_track.name, spotify_name),
        similarity(normalize_title(apple_track.name), normalize_title(spotify_name)),
    )
    artist_score = max(
        similarity(apple_track.artist, spotify_artists),
        similarity(apple_track.album_artist, spotify_artists),
    )
    album_score = similarity(apple_track.album, spotify_album)
    duration_score = 1.0
    spotify_duration = spotify_item.get("duration_ms")
    if apple_track.total_time_ms and spotify_duration:
        delta = abs(apple_track.total_time_ms - spotify_duration)
        if delta <= 2_000:
            duration_score = 1.0
        elif delta <= 5_000:
            duration_score = 0.85
        elif delta <= 10_000:
            duration_score = 0.6
        else:
            duration_score = 0.2

    explicit_penalty = 0.0
    if normalize_text(apple_track.name) == normalize_text(spotify_name):
        explicit_penalty += 0.05

    score = (
        name_score * 0.5
        + artist_score * 0.3
        + album_score * 0.1
        + duration_score * 0.1
        + explicit_penalty
    )
    return min(score, 1.0)


def load_track_match_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_track_match_cache(path: Path, cache: dict[str, dict]) -> None:
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _chunk_dicts(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error, ensure_ascii=False)
    if error:
        return str(error)
    return json.dumps(payload, ensure_ascii=False)