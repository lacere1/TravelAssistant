"""
Places Grounder – resolves raw location strings to structured place data
via the Google Places API (Text Search).

Setup
-----
Set the environment variable ``GOOGLE_PLACES_API_KEY`` to your key.
If the key is absent the grounder silently disables itself and extraction
continues with the raw text (the TfL API can usually resolve it anyway).

Returned structure
------------------
Each successful grounding returns a dict:
    {
        "name":              "Wembley Central",
        "formatted_address": "Wembley Central, Wembley HA9 7AD, UK",
        "place_id":          "ChIJ...",
        "lat":               51.5528,
        "lng":               -0.2966,
    }

Caching
-------
Results are cached in-process so repeated lookups (same session) cost no
additional API calls.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# London centre coordinates – used to bias Place Search results
# ---------------------------------------------------------------------------
_LONDON_LAT = 51.5074
_LONDON_LNG = -0.1278
_SEARCH_RADIUS_M = 30_000   # 30 km – covers Greater London

# How long to wait (seconds) between API requests to stay under rate limits
_REQUEST_DELAY = 0.05


class PlacesGrounder:
    """
    Wraps the Google Places Text Search API with in-process caching.

    Attributes:
        available: True only when a valid API key is configured.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GOOGLE_PLACES_API_KEY") or ""
        self._cache: Dict[str, Optional[Dict]] = {}
        self._last_request_ts: float = 0.0

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def ground(self, query: str) -> Optional[Dict]:
        """
        Resolve *query* to a structured place dict, or None if not found.

        Results are cached so repeated calls with the same string are free.

        Args:
            query: Raw location string, e.g. "wembley", "Oxford Circus",
                   "Lavender Avenue, Clapham".

        Returns:
            Dict with keys ``name``, ``formatted_address``, ``place_id``,
            ``lat``, ``lng``; or ``None`` if resolution fails.
        """
        if not self.available:
            return None

        cache_key = query.strip().lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._call_api(query)
        self._cache[cache_key] = result
        return result

    def ground_many(self, *queries: str) -> Dict[str, Optional[Dict]]:
        """Ground multiple location strings in one call, returning a mapping."""
        return {q: self.ground(q) for q in queries}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, query: str) -> Optional[Dict]:
        """Make one Text Search request and return the top result."""
        try:
            import requests
        except ImportError:
            print("[PlacesGrounder] 'requests' library not installed.")
            return None

        # Simple rate-limiting
        elapsed = time.time() - self._last_request_ts
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)

        # Append "London" to bias results toward the city
        search_query = f"{query.strip()} London"

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query":    search_query,
            "location": f"{_LONDON_LAT},{_LONDON_LNG}",
            "radius":   str(_SEARCH_RADIUS_M),
            "key":      self._api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=5)
            self._last_request_ts = time.time()
            data = resp.json()
        except Exception as exc:
            print(f"[PlacesGrounder] API request failed: {exc}")
            return None

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            print(f"[PlacesGrounder] API status: {data.get('status')} – {data.get('error_message', '')}")
            return None

        results = data.get("results", [])
        if not results:
            return None

        top = results[0]
        location = top.get("geometry", {}).get("location", {})

        return {
            "name":              top.get("name", query),
            "formatted_address": top.get("formatted_address", ""),
            "place_id":          top.get("place_id", ""),
            "lat":               location.get("lat"),
            "lng":               location.get("lng"),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_singleton: Optional[PlacesGrounder] = None


def get_grounder() -> PlacesGrounder:
    """Return (and lazily create) the singleton grounder."""
    global _singleton
    if _singleton is None:
        _singleton = PlacesGrounder()
    return _singleton
