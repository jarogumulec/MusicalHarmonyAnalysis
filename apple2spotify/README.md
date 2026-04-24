# apple2spotify

One-off Apple Music → Spotify migration helpers for this workspace.

## What is implemented

- Parse [apple2spotify/apple_music_Knihovna.xml](apple2spotify/apple_music_Knihovna.xml)
- Build an Apple-side inventory JSON
- Read existing Spotify playlists
- Create an approval-first migration plan JSON
- Apply approved playlist creates/updates additively
- Save Apple favorites into Spotify `Liked Songs`
- Cache track matches so repeated runs are faster
- Export Spotify backups to JSON
- Remove duplicate tracks from a Spotify playlist

## Current scope

- Yes: playlists, playlist matching, `Liked Songs`
- No: album save migration, artist follow migration
- No: full destructive sync
- No: Apple smart playlist rules recreation; only current items snapshot
- No: Spotify folder recreation

## Spotify auth

Set these environment variables before running the commands:

- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`
- `SPOTIPY_REDIRECT_URI`

Recommended redirect URI for local use:

- `http://127.0.0.1:8080/callback`

For convenience, copy [apple2spotify/.env.example](apple2spotify/.env.example) to [apple2spotify/.env](apple2spotify/.env) and fill in the values. The CLI loads that file automatically.

Spotify app setup:

1. Go to Spotify for Developers dashboard.
2. Create an app.
3. Add `http://127.0.0.1:8080/callback` as a Redirect URI.
4. Copy Client ID and Client Secret into [apple2spotify/.env](apple2spotify/.env).

## Commands

From the repository root:

1. Build Apple inventory
   - `uv run python -m apple2spotify inspect-apple`

2. Build approval plan
   - `uv run python -m apple2spotify build-plan`

3. Edit [apple2spotify/migration_plan.json](apple2spotify/migration_plan.json)
   - for playlists set `approval` to one of:
     - `approved_create`
     - `approved_update`
     - `skip`
   - for favorites set `approval` to:
     - `approved_save`
     - `skip`

4. Apply approved actions
   - `uv run python -m apple2spotify apply-plan`
   - favorites in batches from newest Apple additions after a cutoff date:
     - `uv run python -m apple2spotify apply-plan --only-favorites --favorites-added-after 2025-04-01 --favorites-Newest-first --favorites-limit 100`
   - merge only tracks added after a cutoff (useful to skip pre-migration tracks):
     - `uv run python -m apple2spotify apply-plan --merge-added-after 2025-04-01`
   - stop processing a playlist early after N consecutive tracks already exist:
     - `uv run python -m apple2spotify apply-plan --stop-after-n-existing 10`

5. Create a Spotify backup
   - `uv run python -m apple2spotify backup-spotify`

6. Remove duplicate tracks from one playlist
   - `uv run python -m apple2spotify dedupe-playlist --playlist-id <spotify_playlist_id>`

## Generated files

- [apple2spotify/apple_inventory.json](apple2spotify/apple_inventory.json)
- [apple2spotify/migration_plan.json](apple2spotify/migration_plan.json)
- [apple2spotify/spotify_track_cache.json](apple2spotify/spotify_track_cache.json)
- [apple2spotify/sync_report.json](apple2spotify/sync_report.json)

## Backup layout

Each backup run creates one timestamped folder under [apple2spotify/spotify_backups](apple2spotify/spotify_backups).

Inside one backup folder:

- `profile.json` — current Spotify profile metadata
- `playlists.json` — playlist inventory and pointers to per-playlist item dumps
- `liked_tracks.json` — saved tracks / `Liked Songs`
- `playlist_items/` — one JSON file per readable playlist
- `backup_errors.json` — playlists Spotify refused to expose in detail

Old backup runs can be deleted safely if you only want to keep the newest snapshot.

## Notes

- Existing Spotify playlists are matched first by normalized exact name, then by conservative fuzzy name similarity.
- Updates are additive only: missing tracks are added, nothing is removed.
- Ambiguous track matches are skipped and written to the sync report.
- Spotify playlist item operations use the current `/playlists/{id}/items` endpoints, not deprecated `/tracks` endpoints.