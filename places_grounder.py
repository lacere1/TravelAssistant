"""
Places Grounder – resolves raw location strings to structured place data
via OpenStreetMap Nominatim search.

Returned structure
------------------
Each successful grounding returns a dict:
    {
        "name":              "Wembley Central",
        "formatted_address": "Wembley Central, Wembley HA9 7AD, United Kingdom",
        "place_id":          "nominatim:123456",
        "lat":               51.5528,
        "lng":               -0.2966,
    }

Caching
-------
Results are cached in-process so repeated lookups (same session) cost no
additional API calls.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# London bounds – used to restrict search results
# ---------------------------------------------------------------------------
_LONDON_BOUNDS = {
    "left": -0.489,   # west lon
    "right": 0.236,   # east lon
    "top": 51.686,    # north lat
    "bottom": 51.28,  # south lat
}

# Nominatim usage policy is strict; keep to ~1 request per second.
_REQUEST_DELAY = 1.05


class PlacesGrounder:
    """
    Wraps OpenStreetMap Nominatim search with in-process caching.

    Attributes:
        available: True when dependency requirements are met.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Keep signature for backward compatibility (api_key is unused now).
        self._cache: Dict[str, Optional[Dict]] = {}
        self._last_request_ts: float = 0.0

    @property
    def available(self) -> bool:
        return True

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
        """Make one Nominatim request and return the top result."""
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

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": search_query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "gb",
            "viewbox": (
                f"{_LONDON_BOUNDS['left']},{_LONDON_BOUNDS['top']},"
                f"{_LONDON_BOUNDS['right']},{_LONDON_BOUNDS['bottom']}"
            ),
            "bounded": 1,
        }
        headers = {
            "User-Agent": "TravelAssistant/1.0 (local app)",
            "Accept-Language": "en-GB,en",
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=6)
            resp.raise_for_status()
            self._last_request_ts = time.time()
            data = resp.json()
        except Exception as exc:
            print(f"[PlacesGrounder] API request failed: {exc}")
            return None

        if not isinstance(data, list) or not data:
            return None

        top = data[0]
        lat_raw = top.get("lat")
        lon_raw = top.get("lon")
        try:
            lat = float(lat_raw) if lat_raw is not None else None
            lng = float(lon_raw) if lon_raw is not None else None
        except (TypeError, ValueError):
            lat, lng = None, None

        # Extract structured address fields from Nominatim addressdetails.
        # These are far more precise than just lat/lng for disambiguation:
        #   suburb       → neighbourhood name  (e.g. "Sudbury", "Kingsbury")
        #   city_district → borough name        (e.g. "London Borough of Brent")
        #   postcode     → full postcode        (e.g. "HA0 2LL")
        addr = top.get("address") or {}
        raw_postcode = addr.get("postcode", "")
        # Postcode prefix = everything before the space: "HA0 2LL" → "HA0"
        postcode_prefix = raw_postcode.split()[0].upper() if raw_postcode else ""

        # Borough: try city_district first, then look for "London Borough of X"
        # or "Royal Borough of X" pattern in the display_name as a fallback.
        borough = addr.get("city_district", "")
        if not borough:
            import re as _re
            m = _re.search(
                r'((?:London|Royal) Borough of [A-Za-z &]+)',
                top.get("display_name", ""),
            )
            if m:
                borough = m.group(1).strip().rstrip(",")

        return {
            "name": top.get("display_name", query).split(",")[0].strip(),
            "formatted_address": top.get("display_name", ""),
            "place_id": f"nominatim:{top.get('place_id', '')}",
            "lat": lat,
            "lng": lng,
            # Structured address fields — used for hard-boundary candidate matching
            "suburb":           addr.get("suburb", ""),
            "borough":          borough,
            "postcode_prefix":  postcode_prefix,
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
