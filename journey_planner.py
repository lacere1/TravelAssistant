"""
Journey planner backend integrated into the main app.

This file brings over the TfL journey-planning chatbot, its lightweight
datetime NLP, and the TfL Journey API wrapper from the separate
`seperateAdd` project so they can be used inside the current app.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple

import os
import re
import copy
import requests

from disambiguation_engine import (
    DisambiguationCandidate,
    DisambiguationEngine,
    SpatialAnchor,
    JourneyDisambiguationContext,
    JourneyUserHistory,
    NearAreaStructured,
    candidates_from_journey_options,
    journey_disambiguate,
    haversine_km,
)
from places_grounder import get_grounder
from journey_slot_extractor import get_extractor as get_journey_slot_extractor, _extract_near_area

# Microsoft Recognizers-Text has been removed — date/time extraction from
# natural language is now handled by the LLM entity extractor.  The
# lightweight pattern-based parser below is kept as a fallback for
# structured date/time strings (e.g. "tomorrow 9am", "arrive by 6pm").


# ---- Lightweight datetime parser --------------------------------------------


@dataclass
class ParsedWhen:
    date: datetime
    timeIs: str  # "Departing" or "Arriving"


def parse_datetime_text(text: str, now: datetime) -> Optional[dict]:
    """
    Parse natural language date/time phrases like:
      - 'now'
      - 'tomorrow 9am'
      - 'today 18:30'
      - 'arrive by 6pm'

    Returns:
      {
        "datetime": <datetime>,
        "timeIs": "Departing" | "Arriving",
      }
    or None if we can't interpret it.
    """
    if not text or not text.strip():
        return None

    raw = text.strip()
    t = raw.lower()

    # Quick special case for "now"
    if t == "now":
        return {"datetime": now, "timeIs": "Departing"}

    # Lightweight pattern-based parser
    time_is = "Departing"
    if t.startswith("arrive by"):
        time_is = "Arriving"
        t = t.replace("arrive by", "", 1).strip()

    # Very small set of patterns: "tomorrow 9am", "today 18:30", "9am", "18:00"
    day = now.date()
    if t.startswith("tomorrow"):
        day = (now + timedelta(days=1)).date()
        t = t.replace("tomorrow", "", 1).strip()
    elif t.startswith("today"):
        day = now.date()
        t = t.replace("today", "", 1).strip()

    # remaining t should be something like "9am" or "18:30"
    t = t.strip()
    if not t:
        # "tomorrow" alone – assume same time
        dt = datetime.combine(day, now.time())
        return {"datetime": dt, "timeIs": time_is}

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", t)
    if not m:
        return None

    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = m.group(3)

    if ampm:
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0

    try:
        dt = datetime(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=hour,
            minute=minute,
        )
    except ValueError:
        return None

    return {"datetime": dt, "timeIs": time_is}


# ---- TfL Journey API wrapper -----------------------------------------------


TFL_BASE_URL = "https://api.tfl.gov.uk"
OSM_NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"


def _parse_lat_lon_from_id(place_id: str) -> tuple[float | None, float | None]:
    """If place_id is 'lat,lng', return (lat, lon); else (None, None)."""
    if not place_id or "," not in place_id:
        return None, None
    parts = place_id.split(",", 1)
    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except (ValueError, IndexError):
        pass
    return None, None


def _build_place_disambiguation_for_map(
    options: List[Dict[str, Any]], query: str, direction: str
) -> Dict[str, Any]:
    """Build place_disambiguation payload for frontend map (same shape as timetable_disambiguation options)."""
    map_options: List[Dict[str, Any]] = []
    for i, o in enumerate(options):
        lat, lon = _parse_lat_lon_from_id(o.get("id") or "")
        if lat is not None and lon is not None:
            label = o.get("name") or f"{i + 1}"
            map_options.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "label": f"{i + 1}. {o.get('name', '')}" + (f" ({o.get('qualifier', '')})" if o.get("qualifier") else ""),
                    "name": o.get("name", ""),
                }
            )
    return {
        "options": map_options,
        "query": query,
        "direction": direction,
    }


# Greater London bounds for Nominatim's viewbox:
# left, top, right, bottom = west_lon, north_lat, east_lon, south_lat.
LONDON_VIEWBOX = {
    "left": -0.489,
    "top": 51.686,
    "right": 0.236,
    "bottom": 51.28,
}

_PLACE_TYPE_LABELS: Dict[str, str] = {
    "bus_station": "Bus Station",
    "train_station": "Train Station",
    "subway_station": "Underground Station",
    "light_rail_station": "DLR Station",
}

def _stop_type_tag(place_types: List[str]) -> str:
    """Return a short human-readable tag like 'Bus Stop' from place types."""
    for pt in place_types:
        if pt in _PLACE_TYPE_LABELS:
            return _PLACE_TYPE_LABELS[pt]
    if "bus_stop" in place_types:
        return "Bus Stop"
    return ""


def _osm_place_search(query: str, max_results: int = 5, near_area: str | None = None) -> List[Dict[str, Any]]:
    """
    Call OpenStreetMap Nominatim search to find places matching the query.
    Restricted to Greater London via viewbox + bounded search. Returns options
    in the same shape as TfL disambiguation:
    { "id": "lat,lng", "name", "qualifier", "shortLabel", "place_types" }.
    TfL journey API accepts "lat,lng" as from_id/to_id.

    near_area: optional area/neighbourhood hint (e.g. "Sudbury") appended to
    the query so OSM returns geographically biased candidates from the start.
    """
    if not query:
        return []
    # Build enriched query: "Harrow Road, Sudbury, London" rather than just "Harrow Road, London"
    parts = [query]
    if near_area:
        parts.append(near_area)
    if "london" not in query.lower() and (not near_area or "london" not in near_area.lower()):
        parts.append("London")
    search_query = ", ".join(parts)
    params = {
        "q": search_query,
        "format": "jsonv2",
        "addressdetails": 1,
        "countrycodes": "gb",
        "limit": max_results,
        "viewbox": (
            f"{LONDON_VIEWBOX['left']},{LONDON_VIEWBOX['top']},"
            f"{LONDON_VIEWBOX['right']},{LONDON_VIEWBOX['bottom']}"
        ),
        "bounded": 1,
    }
    headers = {
        "User-Agent": "TravelAssistant/1.0 (local app)",
        "Accept-Language": "en-GB,en",
    }
    print(f"[OSM] GET {OSM_NOMINATIM_SEARCH_URL}")
    print(f"[OSM] Params: {params}")
    try:
        resp = requests.get(OSM_NOMINATIM_SEARCH_URL, params=params, headers=headers, timeout=6)
        print(f"[OSM] Status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[OSM] FAILED: {e}")
        return []
    places = data if isinstance(data, list) else []
    print(f"[OSM] Returned {len(places)} places:")
    for idx, p in enumerate(places):
        name = (p.get("display_name") or "?").split(",")[0].strip()
        addr = p.get("display_name") or ""
        lat = p.get("lat", "?")
        lon = p.get("lon", "?")
        p_class = p.get("class", "")
        p_type = p.get("type", "")
        print(f"[OSM]   {idx+1}. {name} | {addr} | ({lat}, {lon}) | class={p_class} type={p_type}")
    results: List[Dict[str, Any]] = []
    for p in places:
        lat_raw = p.get("lat")
        lon_raw = p.get("lon")
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            continue
        place_id = f"{lat},{lon}"
        display_name = p.get("display_name") or query
        name = display_name.split(",")[0].strip() or query
        p_class = (p.get("class") or "").strip()
        p_type = (p.get("type") or "").strip()
        types = [v for v in (p_class, p_type) if v]
        addr = display_name
        results.append(
            {
                "id": place_id,
                "name": name,
                "qualifier": addr,
                "shortLabel": name,
                "place_types": types,
            }
        )
    return results


# London centre — bias for Google Places Text Search
_GOOGLE_PLACES_LOCATION_BIAS = "51.5074,-0.1278"
_GOOGLE_PLACES_RADIUS_M = 45000

# Merge OSM + Google hits that are the same road/place under different postcodes
_COALESCE_MAX_DISTANCE_KM = 0.75
_COALESCE_NAME_RATIO = 0.90
# How many ranked candidates to show when asking for journey origin/destination
_JOURNEY_DISAMBIG_UI_MAX = 5
# Minimum query↔name similarity (SequenceMatcher ratio) to list a candidate
_JOURNEY_NAME_MATCH_MIN_DISPLAY = 0.35


def _journey_name_match_passes(c: DisambiguationCandidate) -> bool:
    """True if candidate should be shown (or auto-selected) on name similarity alone."""
    ns = getattr(c, "name_similarity", None)
    if ns is None:
        return True
    try:
        return float(ns) >= _JOURNEY_NAME_MATCH_MIN_DISPLAY
    except (TypeError, ValueError):
        return True


def _option_lat_lng(o: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Read coordinates from explicit fields or from a 'lat,lng' id."""
    lat = o.get("lat")
    lng = o.get("lng")
    if lng is None:
        lng = o.get("lon")
    if lat is not None and lng is not None:
        try:
            return float(lat), float(lng)
        except (TypeError, ValueError):
            pass
    oid = str(o.get("id") or "")
    if "," in oid:
        parts = oid.split(",", 1)
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except (TypeError, ValueError):
            pass
    return None, None


def _dedupe_exact_coord_options(options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate results that share the same id (e.g. same lat,lng from two APIs)."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for o in options:
        lid = str(o.get("id") or "").strip()
        if not lid:
            continue
        if lid in seen:
            continue
        seen.add(lid)
        out.append(o)
    return out


def _pick_representative_index(
    members: List[int],
    options: List[Dict[str, Any]],
    norms: List[str],
    q_norm: str,
) -> int:
    """Prefer higher query–name similarity, then Google over OSM when tied."""
    best_idx = members[0]
    best_key: Optional[Tuple[float, int]] = None
    for idx in members:
        sim = SequenceMatcher(None, q_norm, norms[idx]).ratio()
        src_boost = 1 if options[idx].get("_source") == "google" else 0
        key = (sim, src_boost)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    return best_idx


def _coalesce_near_duplicate_options(
    options: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """
    Group results that refer to the same feature: very similar normalised names
    and within _COALESCE_MAX_DISTANCE_KM. Keeps one representative per cluster.
    """
    if len(options) <= 1:
        out = []
        q_norm = DisambiguationEngine._normalize_place_name(query)
        for o in options:
            oc = dict(o)
            n = DisambiguationEngine._normalize_place_name(oc.get("name") or "")
            oc["name_similarity"] = round(SequenceMatcher(None, q_norm, n).ratio(), 4)
            oc.pop("_source", None)
            out.append(oc)
        return out

    n = len(options)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    coords = [_option_lat_lng(o) for o in options]
    norms = [DisambiguationEngine._normalize_place_name(o.get("name") or "") for o in options]

    for i in range(n):
        for j in range(i + 1, n):
            if coords[i][0] is None or coords[j][0] is None:
                continue
            if SequenceMatcher(None, norms[i], norms[j]).ratio() < _COALESCE_NAME_RATIO:
                continue
            d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if d <= _COALESCE_MAX_DISTANCE_KM:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    q_norm = DisambiguationEngine._normalize_place_name(query)
    merged: List[Dict[str, Any]] = []
    for members in clusters.values():
        best_idx = _pick_representative_index(members, options, norms, q_norm)
        rep = dict(options[best_idx])
        rep["name_similarity"] = round(
            SequenceMatcher(None, q_norm, norms[best_idx]).ratio(),
            4,
        )
        if len(members) > 1:
            print(
                f"[JourneyMerge] Coalesced {len(members)} options → "
                f"'{rep.get('name')}' @ {rep.get('id')}"
            )
        rep.pop("_source", None)
        merged.append(rep)

    merged.sort(key=lambda x: x.get("name_similarity") or 0.0, reverse=True)
    return merged


def _google_places_text_search(query: str, max_results: int = 20, near_area: str | None = None) -> List[Dict[str, Any]]:
    """
    Google Places Text Search (legacy) — returns options in the same shape as OSM search.
    TfL journey API accepts "lat,lng" as from_id/to_id.

    near_area: optional area hint appended to the query for geographic bias.
    """
    if not query:
        return []
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if not api_key:
        return []
    parts = [query]
    if near_area:
        parts.append(near_area)
    base = ", ".join(parts)
    search_query = base if "london" in base.lower() else f"{base}, London, UK"
    params = {
        "query": search_query,
        "key": api_key,
        "region": "uk",
        "location": _GOOGLE_PLACES_LOCATION_BIAS,
        "radius": _GOOGLE_PLACES_RADIUS_M,
    }
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    print(f"[Google Places] GET textsearch (query={search_query!r})")
    try:
        resp = requests.get(url, params=params, timeout=8)
        print(f"[Google Places] Status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Google Places] FAILED: {e}")
        return []
    status = str(data.get("status") or "")
    if status not in ("OK", "ZERO_RESULTS"):
        print(f"[Google Places] API status={status}; message={data.get('error_message')!r}")
        return []
    places = data.get("results") or []
    if not isinstance(places, list):
        return []
    results: List[Dict[str, Any]] = []
    for p in places[:max_results]:
        loc = (p.get("geometry") or {}).get("location") or {}
        lat_raw = loc.get("lat")
        lng_raw = loc.get("lng")
        try:
            lat = float(lat_raw)
            lng = float(lng_raw)
        except (TypeError, ValueError):
            continue
        place_id = f"{lat},{lng}"
        name = (p.get("name") or "").strip() or query
        addr = (p.get("formatted_address") or "").strip()
        types = [str(t).replace("_", " ") for t in (p.get("types") or []) if t]
        results.append(
            {
                "id": place_id,
                "name": name,
                "qualifier": addr or ", ".join(types[:3]) if types else "",
                "shortLabel": name,
                "place_types": types,
            }
        )
    print(f"[Google Places] Returned {len(results)} results")
    return results


def _fetch_merged_journey_location_options(query: str, max_results: int = 20, near_area: str | None = None) -> List[Dict[str, Any]]:
    """
    Fetch from both Google Places Text Search and OSM Nominatim, dedupe by id,
    then coalesce near-duplicate names in the same area.

    near_area: optional area hint (e.g. "Sudbury") passed to both APIs so
    results are geographically biased from the start.
    """
    google_opts = _google_places_text_search(query, max_results=max_results, near_area=near_area)
    for o in google_opts:
        o["_source"] = "google"
    osm_opts = _osm_place_search(query, max_results=max_results, near_area=near_area)
    for o in osm_opts:
        o["_source"] = "osm"
    combined = _dedupe_exact_coord_options(google_opts + osm_opts)
    print(
        f"[Journey] Merged search: Google={len(google_opts)} OSM={len(osm_opts)} "
        f"after id-dedupe={len(combined)}"
    )
    return _coalesce_near_duplicate_options(combined, query)


@dataclass
class JourneyLeg:
    mode: str
    detail: str


@dataclass
class JourneySummary:
    departure: str  # "HH:MM"
    arrival: str  # "HH:MM"
    duration: int  # minutes
    legs: List[JourneyLeg]


class TflJourneyClient:
    """
    Thin wrapper around TfL Journey Planner endpoints.
    You will need to set TFL_APP_ID and TFL_APP_KEY as environment variables.
    """

    def __init__(self):
        self.session = requests.Session()
        self.app_id = os.getenv("TFL_APP_ID")
        self.app_key = os.getenv("TFL_APP_KEY")
        # Last TfL URL used for a request (for debugging/inspection).
        self.last_url: str | None = None

    # ---- Helpers --------------------------------------------------------

    # Mapping from commonly used / LLM-output mode names to the actual
    # identifiers accepted by the TfL Journey Planner API.
    # Source: https://api.tfl.gov.uk/Journey/Meta/Modes
    _MODE_ALIASES: Dict[str, str] = {
        # Correct API identifiers (pass through)
        "bus": "bus",
        "tube": "tube",
        "overground": "overground",
        "dlr": "dlr",
        "tram": "tram",
        "coach": "coach",
        "walking": "walking",
        "cycle": "cycle",
        "national-rail": "national-rail",
        "elizabeth-line": "elizabeth-line",
        "cable-car": "cable-car",
        "river-bus": "river-bus",
        "cycle-hire": "cycle-hire",
        "interchange-keep-sitting": "interchange-keep-sitting",
        "interchange-secure": "interchange-secure",
        # Common aliases from old documentation / LLM output
        "public-bus": "bus",
        "train": "national-rail",
        "rail": "national-rail",
        "cablecar": "cable-car",
        "cable car": "cable-car",
        "river": "river-bus",
        "riverbus": "river-bus",
        "river bus": "river-bus",
        "elizabeth": "elizabeth-line",
        "crossrail": "elizabeth-line",
        "walk": "walking",
        "bike": "cycle",
        "bicycle": "cycle",
    }

    @staticmethod
    def _build_extra_params(
        via: str | None = None,
        mode: str | None = None,
        journey_preference: str | None = None,
    ) -> Dict[str, str]:
        """
        Build the optional TfL query parameters from LLM-extracted entities,
        normalising casing and format to match what the TfL API expects.

        TfL Journey API parameter reference:
          via              – free-text, postcode, Naptan StopPoint ID, ICS StopId,
                             or WGS84 "lat,long"
          mode             – comma-separated list from Journey/Meta/Modes:
                             bus, tube, national-rail, overground, dlr,
                             elizabeth-line, tram, cable-car, river-bus,
                             coach, walking, cycle, cycle-hire
          journeyPreference – LeastInterchange | LeastTime | LeastWalking
          timeIs           – Departing | Arriving  (handled separately via `when`)
        """
        extra: Dict[str, str] = {}

        if via:
            extra["via"] = via.strip()

        if mode:
            # Normalise each mode token through the alias map, dropping unknowns
            raw_modes = [m.strip().lower() for m in mode.split(",") if m.strip()]
            mapped = []
            for m in raw_modes:
                canonical = TflJourneyClient._MODE_ALIASES.get(m)
                if canonical:
                    if canonical not in mapped:
                        mapped.append(canonical)
                else:
                    print(f"[TfL] WARNING: unknown mode '{m}' — dropping from request")
            if mapped:
                extra["mode"] = ",".join(mapped)
                print(f"[TfL] Normalised modes: {raw_modes} → {mapped}")

        if journey_preference:
            # TfL expects PascalCase: LeastInterchange, LeastTime, LeastWalking
            _JP_MAP = {
                "leastinterchange": "LeastInterchange",
                "leasttime": "LeastTime",
                "leastwalking": "LeastWalking",
            }
            jp_lower = journey_preference.strip().lower()
            extra["journeyPreference"] = _JP_MAP.get(jp_lower, journey_preference)

        if extra:
            print(f"[TfL] Extra API params: {extra}")
        return extra

    @staticmethod
    def _extract_via_disambiguation(data: Dict[str, Any]) -> str | None:
        """
        If the TfL response contains a viaLocationDisambiguation with options,
        return the parameterValue of the first (best) match so we can retry
        with a resolved via ID instead of free text.
        """
        via_disamb = data.get("viaLocationDisambiguation") or {}
        if via_disamb.get("matchStatus") not in ("list", "identified"):
            return None
        options = via_disamb.get("disambiguationOptions") or []
        if not options:
            return None
        first = options[0]
        param_val = first.get("parameterValue", "")
        place_name = (first.get("place") or {}).get("commonName", "")
        if param_val:
            print(f"[TfL] Via disambiguation: auto-resolving to '{place_name}' (id={param_val})")
            return param_val
        return None

    def _try_journey_request(
        self,
        url: str,
        params: Dict[str, str],
        label: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Make a single journey API request, handling:
          - Normal 200 responses with journeys
          - 300 via-disambiguation responses (auto-resolves and retries once)
          - Errors / empty results (returns [])
        """
        try:
            resp = self.session.get(url, params=params, timeout=8)
            print(f"[TfL] {label} status: {resp.status_code}")

            # On 4xx/5xx, raise immediately so the caller can try a fallback
            if resp.status_code >= 400:
                resp.raise_for_status()

            data = resp.json()

            # Check for from/to disambiguation (log only, don't auto-resolve)
            for side in ("from", "to"):
                disamb = data.get(f"{side}LocationDisambiguation") or {}
                if disamb.get("matchStatus") == "list":
                    print(f"[TfL] {label}: '{side}' needs disambiguation")

            journeys = data.get("journeys") or []
            print(f"[TfL] {label}: {len(journeys)} journeys returned")

            if journeys:
                return _journeys_to_summaries(journeys)

            # No journeys — check if via needs disambiguation and retry once
            if "via" in params:
                resolved_via = self._extract_via_disambiguation(data)
                if resolved_via:
                    retry_params = dict(params)
                    retry_params["via"] = resolved_via
                    print(f"[TfL] {label}: retrying with resolved via={resolved_via}")
                    try:
                        resp2 = self.session.get(url, params=retry_params, timeout=8)
                        print(f"[TfL] {label} via-retry status: {resp2.status_code}")
                        if resp2.status_code < 400:
                            data2 = resp2.json()
                            journeys2 = data2.get("journeys") or []
                            print(f"[TfL] {label} via-retry: {len(journeys2)} journeys")
                            if journeys2:
                                return _journeys_to_summaries(journeys2)
                    except Exception as exc2:
                        print(f"[TfL] {label} via-retry failed: {exc2}")

            return []
        except Exception as exc:
            print(f"[TfL] {label} failed: {exc}")
            return []

    # ---- Public methods -------------------------------------------------

    def disambiguate_location(self, query: str) -> List[Dict[str, Any]]:
        """
        Calls the journey results endpoint with a free-text location to let TfL
        perform disambiguation. Returns a list of options with id/name/qualifier.
        """
        if not query:
            return []

        params = self._auth_params()
        # Using the same URL pattern as in the Westminster→Bank example.
        url = f"{TFL_BASE_URL}/journey/journeyresults/{requests.utils.quote(query)}/to/bank"
        self.last_url = url

        try:
            resp = self.session.get(url, params=params, timeout=5)
            resp.raise_for_status()
        except Exception:
            return []

        data = resp.json()

        # The real API returns "fromLocationDisambiguation" / "toLocationDisambiguation"
        # structures when it can't uniquely identify the stop. Here we only look at "from".
        from_disamb = data.get("fromLocationDisambiguation") or {}
        disambiguation_options = from_disamb.get("disambiguationOptions") or []

        options: List[Dict[str, Any]] = []
        for option in disambiguation_options:
            place = option.get("place", {})
            parameter_value = option.get("parameterValue", "")
            common_name = place.get("commonName", query)
            place_type = place.get("placeType", "")

            # Use parameterValue as the ID (it can be an ICS code like "1000100" or coordinates)
            if not parameter_value:
                continue

            options.append(
                {
                    "id": parameter_value,
                    "name": common_name,
                    "qualifier": place_type,
                    "shortLabel": common_name,
                }
            )

        return options

    def search_places(self, query: str) -> List[Dict[str, Any]]:
        """
        Use TfL's Place Search endpoint for autocomplete-style location lookup.
        Returns a list of options with id/name/shortLabel suitable for UI use.
        """
        if not query:
            return []

        params = self._auth_params()
        # TfL Place Search uses `name` as the query parameter and returns
        # a JSON array of Place objects (as seen in the Waterloo example).
        params["name"] = query
        url = f"{TFL_BASE_URL}/Place/Search"
        self.last_url = url

        try:
            resp = self.session.get(url, params=params, timeout=5)
            resp.raise_for_status()
        except Exception:
            return []

        places = resp.json()

        results: List[Dict[str, Any]] = []
        for p in places:
            pid = p.get("id") or p.get("naptanId")
            if not pid:
                continue
            name = p.get("name") or p.get("commonName") or query
            place_type = p.get("placeType") or ",".join(p.get("modes", []))
            results.append(
                {
                    "id": pid,
                    "name": name,
                    "qualifier": place_type,
                    "shortLabel": name,
                }
            )

        return results

    def get_journeys(
        self,
        from_id: str,
        to_id: str,
        when: Dict[str, Any],
        via: str | None = None,
        mode: str | None = None,
        journey_preference: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetches journeys between two resolved place IDs around a given datetime.

        Parameters
        ----------
        from_id             : Resolved TfL place ID / coordinate / postcode for origin.
        to_id               : Resolved TfL place ID / coordinate / postcode for destination.
        when                : Dict with "datetime" (datetime obj) and "timeIs" ("Departing"|"Arriving").
        via                 : Optional intermediate waypoint (free-text, postcode, Naptan ID, or "lat,lon").
        mode                : Optional comma-separated TfL transport modes (TfL identifiers).
        journey_preference  : Optional journey optimisation: "leastinterchange"|"leasttime"|"leastwalking".
        """
        dt: datetime = when["datetime"]
        time_is: str = when["timeIs"]  # "Departing" or "Arriving"

        url = f"{TFL_BASE_URL}/journey/journeyresults/{from_id}/to/{to_id}"
        self.last_url = url

        extra_params = self._build_extra_params(via, mode, journey_preference)

        # Primary call: TfL format – date=YYYYMMDD, time=HHmm, timeIs=Departing|Arriving.
        params = self._auth_params()
        params.update({"date": dt.strftime("%Y%m%d"), "time": dt.strftime("%H%M"), "timeIs": time_is})
        params.update(extra_params)
        print(f"[TfL] get_journeys URL: {url}")

        result = self._try_journey_request(url, params, label="get_journeys")
        if result:
            return result

        # Fallback: date/time with hyphen/colon.
        params_alt = self._auth_params()
        params_alt.update({"date": dt.strftime("%Y-%m-%d"), "time": dt.strftime("%H:%M"), "timeIs": time_is})
        params_alt.update(extra_params)
        result2 = self._try_journey_request(url, params_alt, label="get_journeys-fb1")
        if result2:
            return result2

        # Fallback: without date/time.
        params_fb = self._auth_params()
        params_fb.update(extra_params)
        return self._try_journey_request(url, params_fb, label="get_journeys-fb2")

    def get_journeys_by_queries(
        self,
        from_query: str,
        to_query: str,
        when: Dict[str, Any],
        via: str | None = None,
        mode: str | None = None,
        journey_preference: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Tries the journey API with from/to as place names (e.g. "Royal Borough of Kingston upon Thames").
        If TfL returns itineraries directly, returns them; otherwise returns [] so the caller can disambiguate.

        Parameters
        ----------
        via                 : Optional intermediate waypoint (free-text, postcode, Naptan ID, or "lat,lon").
        mode                : Optional comma-separated TfL transport modes.
        journey_preference  : Optional journey optimisation: "leastinterchange"|"leasttime"|"leastwalking".
        """
        if not from_query.strip() or not to_query.strip():
            return []
        dt: datetime = when["datetime"]
        time_is: str = when["timeIs"]
        url = f"{TFL_BASE_URL}/journey/journeyresults/{requests.utils.quote(from_query.strip())}/to/{requests.utils.quote(to_query.strip())}"
        self.last_url = url

        extra_params = self._build_extra_params(via, mode, journey_preference)
        print(f"[TfL] get_journeys_by_queries URL: {url}")

        # Primary call: TfL format – date=YYYYMMDD, time=HHmm, timeIs=Departing|Arriving.
        params = self._auth_params()
        params.update({"date": dt.strftime("%Y%m%d"), "time": dt.strftime("%H%M"), "timeIs": time_is})
        params.update(extra_params)

        result = self._try_journey_request(url, params, label="by_queries")
        if result:
            return result

        # Fallback: date/time with hyphen and colon.
        params_alt = self._auth_params()
        params_alt.update({"date": dt.strftime("%Y-%m-%d"), "time": dt.strftime("%H:%M"), "timeIs": time_is})
        params_alt.update(extra_params)
        result2 = self._try_journey_request(url, params_alt, label="by_queries-fb1")
        if result2:
            return result2

        # Fallback without date/time.
        params_fb = self._auth_params()
        params_fb.update(extra_params)
        return self._try_journey_request(url, params_fb, label="by_queries-fb2")

    # ---- Internal helpers -----------------------------------------------

    def _auth_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if self.app_id and self.app_key:
            params["app_id"] = self.app_id
            params["app_key"] = self.app_key
        return params


def _journeys_to_summaries(journeys: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse TfL journey list into our summary format."""
    summaries: List[Dict[str, Any]] = []
    for j in journeys:
        start_dt = _parse_tfl_iso(j.get("startDateTime"))
        end_dt = _parse_tfl_iso(j.get("arrivalDateTime"))
        duration = j.get("duration")

        legs_summary: List[Dict[str, Any]] = []
        for leg in j.get("legs", []):
            mode = leg.get("mode", {}).get("name", "").title()
            instr_obj = leg.get("instruction", {})
            instr = instr_obj.get("summary", "")
            line_name = (
                leg.get("routeOptions", [{}])[0]
                .get("lineIdentifier", {})
                .get("name", "")
            )
            leg_duration = leg.get("duration")
            if leg_duration is None:
                leg_duration = 0

            detail_parts = []
            if line_name:
                detail_parts.append(line_name)
            if instr:
                detail_parts.append(instr)

            leg_out: Dict[str, Any] = {
                "mode": mode or "Walk",
                "detail": " – ".join(detail_parts) if detail_parts else "",
                "duration": leg_duration,
            }

            # Walking: extract step-by-step instructions and path for directions URL
            raw_mode = (leg.get("mode", {}).get("name") or "").lower()
            if raw_mode in ("walk", "walking"):
                steps = instr_obj.get("steps", [])
                leg_out["steps"] = [
                    {
                        "description": s.get("description")
                        or s.get("detailedDescription")
                        or s.get("turnInstruction", ""),
                        "distance": s.get("distance") or 0,
                    }
                    for s in (steps if isinstance(steps, list) else [])
                ]
                path = leg.get("path", {}) or {}
                line_str = path.get("lineString", "")
                if line_str:
                    points = [p.strip() for p in line_str.split(" ") if p.strip()]
                    if len(points) >= 2:
                        leg_out["fromLatLng"] = points[0]
                        leg_out["toLatLng"] = points[-1]
            else:
                # Transit: extract stop names for "View stops"
                path = leg.get("path", {}) or {}
                stop_points = path.get("stopPoints", []) or []
                leg_out["stops"] = []
                for sp in stop_points:
                    if isinstance(sp, dict):
                        name = (
                            sp.get("name")
                            or sp.get("commonName")
                            or sp.get("stationName")
                            or ""
                        )
                        if name:
                            leg_out["stops"].append(name)
                    elif isinstance(sp, str):
                        leg_out["stops"].append(sp)

            legs_summary.append(leg_out)

        fare_total = None
        fare = j.get("fare") or {}
        if fare.get("totalCost") is not None:
            fare_total = fare["totalCost"]

        if not start_dt or not end_dt or duration is None:
            continue

        summaries.append(
            {
                "departure": start_dt.strftime("%H:%M"),
                "arrival": end_dt.strftime("%H:%M"),
                "duration": duration,
                "legs": legs_summary,
                "fare_pence": fare_total,
            }
        )
    return summaries


def _parse_tfl_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # TfL uses ISO 8601; Python 3.11+ has fromisoformat supporting this reasonably well.
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ---- Journey-planner chatbot -----------------------------------------------


def _parse_journey_datetime(text: str, now: datetime) -> Optional[Dict[str, Any]]:
    """
    Extract date and time from journey phrases (e.g. "at 6:00", "at 6:00pm",
    "at 6am", "on 2026-03-10 at 6:00", "tomorrow at 9am"). Used when parsing
    "plan a journey from X to Y at 6:00". Pattern-based extraction; if this
    cannot parse a date/time, the caller assumes NOW. Returns when dict or None.
    """
    text = text.strip()
    if not text:
        return None

    # Try parse_datetime_text first (handles "now", "tomorrow 9am", "arrive by 6pm")
    parsed = parse_datetime_text(text, now=now)
    if parsed:
        return parsed

    # " on YYYY-MM-DD at H:MM" or " at H:MM" anywhere in text
    m = re.search(
        r"\s+on\s+(\d{4}-\d{2}-\d{2})\s+at\s+(\d{1,2}:\d{2})\s*", text, re.IGNORECASE
    )
    if m:
        try:
            dt = datetime.strptime(
                m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M"
            )
            return {"datetime": dt, "timeIs": "Departing"}
        except ValueError:
            pass

    # " at 6:00" or " at 6:00pm" or " at 6am" or " at 18:00" (with optional am/pm)
    time_patterns = [
        (r"\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)?\s*$", lambda g: (int(g[1]), int(g[2]), g[3])),
        (r"\s+at\s+(\d{1,2})\s*(am|pm)\s*$", lambda g: (int(g[1]), 0, g[2])),
        (r"\s+at\s+(\d{1,2}):(\d{2})\s*$", lambda g: (int(g[1]), int(g[2]), None)),
        (r"\s+at\s+(\d{1,2})\s*$", lambda g: (int(g[1]), 0, None)),
    ]
    for pattern, get_hm in time_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                from datetime import date

                hour, minute, ampm = get_hm(m)
                if ampm:
                    if ampm and ampm.lower() == "pm" and hour < 12:
                        hour += 12
                    if ampm and ampm.lower() == "am" and hour == 12:
                        hour = 0
                if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                    continue
                d = date.today()
                dt = datetime(d.year, d.month, d.day, hour, minute)
                return {"datetime": dt, "timeIs": "Departing"}
            except (ValueError, IndexError):
                continue

    return None


def _strip_datetime_suffix(text: str) -> str:
    """
    Remove trailing date/time phrases from a string so we get just the place name.
    E.g. "Willesden at 6:00" -> "Willesden", "X on 2026-03-10 at 18:00" -> "X".
    """
    t = text.strip()
    # Strip " on YYYY-MM-DD at H:MM" or " at H:MM" or " at 6am" etc.
    t = re.sub(
        r"\s+on\s+\d{4}-\d{2}-\d{2}\s+at\s+\d{1,2}(?::\d{2})?\s*(am|pm)?\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\s+at\s+\d{1,2}(?::\d{2})?\s*(am|pm)?\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # Also strip phrases like "at tomorrow 6pm" or "tomorrow at 6pm"
    t = re.sub(
        r"\s+at\s+(today|tomorrow)\s+\d{1,2}(?::\d{2})?\s*(am|pm)?\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\s+(today|tomorrow)\s+at\s+\d{1,2}(?::\d{2})?\s*(am|pm)?\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t.strip()


def _parse_on_date_time(text: str) -> Dict[str, Any] | None:
    """If text ends with ' on YYYY-MM-DD at H:MM' or ' at H:MM', parse and return when dict."""
    text = text.strip()
    m = re.search(
        r"\s+on\s+(\d{4}-\d{2}-\d{2})\s+at\s+(\d{1,2}:\d{2})\s*$", text, re.IGNORECASE
    )
    if m:
        try:
            dt = datetime.strptime(
                m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M"
            )
            return {"datetime": dt, "timeIs": "Departing"}
        except ValueError:
            pass
    m = re.search(r"\s+at\s+(\d{1,2}:\d{2})\s*$", text, re.IGNORECASE)
    if m:
        try:
            from datetime import date

            t = datetime.strptime(m.group(1), "%H:%M").time()
            d = date.today()
            return {"datetime": datetime.combine(d, t), "timeIs": "Departing"}
        except ValueError:
            pass
    return None


class JourneyChatbot:
    """
    Very simple, stateful dialogue manager for a TfL journey-planning assistant.
    In a real app you would keep user_state in a DB or server-side session.
    Here we keep a single in-memory state dictionary keyed by 'global'.
    Uses OpenStreetMap Nominatim for place disambiguation.
    """

    def __init__(self):
        self.tfl_client = TflJourneyClient()
        self.state: Dict[str, Dict[str, Any]] = {"global": {}}

    def reset(self) -> None:
        """Reset all journey-planning state for a fresh conversation."""
        self.state["global"].clear()

    @staticmethod
    def _looks_like_compound_journey_text(q: str) -> bool:
        """True if *q* looks like a full journey sentence, not a single place name."""
        if not (q or "").strip():
            return False
        sl = q.lower()
        if "plan" in sl and "journey" in sl:
            return True
        # Spaced " from " / " to " — not e.g. "frome" as a town name at start
        return " from " in f" {q} " and " to " in sl

    def _sanitize_compound_location_queries(
        self, user_state: Dict[str, Any], raw_message: str
    ) -> None:
        """
        When the LLM puts an entire utterance into origin (or similar), OSM search
        receives a useless query and returns []. Recover from/to via the same
        rule-based extractor used when the LLM is disabled.
        """
        fq = (user_state.get("fromQuery") or "").strip()
        tq = (user_state.get("toQuery") or "").strip()
        if not self._looks_like_compound_journey_text(fq) and not self._looks_like_compound_journey_text(
            tq
        ):
            return
        msg = (raw_message or "").strip()
        if not msg:
            return
        try:
            slots = get_journey_slot_extractor().extract_journey_slots(msg)
        except Exception as exc:
            print(f"[JourneyChatbot] Compound slot recovery failed: {exc}")
            return
        if slots.get("origin"):
            user_state["fromQuery"] = slots["origin"]
            print(f"[JourneyChatbot] Sanitized fromQuery → {slots['origin']!r}")
            if slots.get("origin_near_area"):
                user_state["_nlp_origin_near_area"] = slots["origin_near_area"]
                print(f"[JourneyChatbot] Slot extractor origin_near_area → {slots['origin_near_area']!r}")
        if slots.get("destination"):
            user_state["toQuery"] = slots["destination"]
            print(f"[JourneyChatbot] Sanitized toQuery → {slots['destination']!r}")
            if slots.get("destination_near_area"):
                user_state["_nlp_destination_near_area"] = slots["destination_near_area"]
                print(f"[JourneyChatbot] Slot extractor destination_near_area → {slots['destination_near_area']!r}")

    # ---- Public API -----------------------------------------------------

    def handle_message(
        self,
        text: str,
        now: datetime,
        username: str | None = None,
        nlp_origin: str | None = None,
        nlp_destination: str | None = None,
        nlp_entities: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Core entry point called from Flask.

        Parameters
        ----------
        text            : Raw user message.
        now             : Current UTC datetime.
        username        : Optional display name for personalised replies.
        nlp_origin      : Origin extracted by the NLP pipeline (used as
                          fallback when text-based extraction finds nothing).
        nlp_destination : Destination extracted by the NLP pipeline.
        nlp_entities    : Full entity dict from the NLP/LLM pipeline,
                          containing via, mode, timeIs, journey_preference, etc.

        Returns a JSON structure the frontend can render:
        {
          "reply": "...",
          "journeys": [...],     # optional, structured results
          "state": {...}         # debug/inspection if desired
        }
        """
        user_state = self.state["global"]

        # Store LLM-extracted journey parameters in user_state so they
        # propagate through to the TfL API call in _continue_planning.
        if nlp_entities:
            for key in ("via", "mode", "journey_preference", "timeIs", "near_area", "towards", "origin_near_area", "destination_near_area"):
                if key in nlp_entities and nlp_entities[key]:
                    user_state[f"_nlp_{key}"] = nlp_entities[key]
                    print(f"[JourneyChatbot] Stored _nlp_{key}={nlp_entities[key]!r}")

        # 0) If we previously asked the user to "rephrase or give a nearby
        # station or area" and stored which side ('from' or 'to') failed, then
        # treat this new message as the updated location for that side.
        pending_side = user_state.get("pending_location_rephrase_for")
        if pending_side in ("from", "to") and (text or "").strip():
            key = f"{pending_side}Query"
            user_state[key] = text.strip()
            # Clear any stale resolution/disambiguation state so we start afresh.
            for k in (
                f"{pending_side}LocationId",
                f"{pending_side}Options",
                f"{pending_side}Question",
            ):
                user_state.pop(k, None)
            user_state.pop("pending_location_rephrase_for", None)
            user_state["journey_planning_active"] = True
            # Old "near X" anchor from the failed query must not skew the new search.
            user_state.pop("_nlp_near_area", None)
            user_state.pop("_nlp_origin_near_area", None)
            user_state.pop("_nlp_destination_near_area", None)
            user_state.pop("_nlp_towards", None)
            # Continue planning immediately with the updated query.
            return self._continue_planning(text, user_state, now, username=username)

        # 1) LLM-extracted slots are the sole source for journey origin/destination.
        #    Rule-based regex extraction (_apply_full_plan_if_present,
        #    _maybe_extract_initial_intent) has been removed.
        if nlp_origin and "fromQuery" not in user_state:
            # Strip any residual "near X" from LLM origin (belt-and-braces)
            origin_clean, origin_area = _extract_near_area(nlp_origin)
            user_state["fromQuery"] = origin_clean
            if origin_area and not user_state.get("_nlp_origin_near_area"):
                user_state["_nlp_origin_near_area"] = origin_area
                print(f"[JourneyChatbot] Extracted origin_near_area from LLM origin: {origin_area!r}")
            print(f"[JourneyChatbot] fromQuery from LLM: {origin_clean!r}")
        if nlp_destination and "toQuery" not in user_state:
            # Strip any residual "near X" from LLM destination (belt-and-braces)
            dest_clean, dest_area = _extract_near_area(nlp_destination)
            user_state["toQuery"] = dest_clean
            if dest_area and not user_state.get("_nlp_destination_near_area"):
                user_state["_nlp_destination_near_area"] = dest_area
                print(f"[JourneyChatbot] Extracted destination_near_area from LLM dest: {dest_area!r}")
            print(f"[JourneyChatbot] toQuery from LLM: {dest_clean!r}")

        # If we're still missing a resolved place id (e.g. OSM failed and pending_* was
        # lost) but the LLM extracted a new place for that side, prefer it over the
        # stale query. Skip when we're choosing from numbered options or pin-drop.
        if not user_state.get("pending_location_rephrase_for"):
            if (
                "fromLocationId" not in user_state
                and not user_state.get("fromOptions")
                and not user_state.get("fromPinDrop")
                and nlp_origin
                and (nlp_origin or "").strip()
                != (user_state.get("fromQuery") or "").strip()
            ):
                user_state["fromQuery"] = (nlp_origin or "").strip()
                user_state.pop("_nlp_near_area", None)
                user_state.pop("_nlp_origin_near_area", None)
                user_state.pop("_nlp_destination_near_area", None)
                user_state.pop("_nlp_towards", None)
            if (
                "toLocationId" not in user_state
                and not user_state.get("toOptions")
                and not user_state.get("toPinDrop")
                and nlp_destination
                and (nlp_destination or "").strip()
                != (user_state.get("toQuery") or "").strip()
            ):
                user_state["toQuery"] = (nlp_destination or "").strip()

        self._sanitize_compound_location_queries(user_state, text)

        # Also extract date/time from LLM entities if available and no when set yet
        if nlp_entities and "when" not in user_state:
            llm_time = nlp_entities.get("time")
            llm_date = nlp_entities.get("date")
            if llm_time or llm_date:
                # Try to parse date/time from LLM output; fall back to "now" on failure
                when = _parse_journey_datetime(text, now=now) or _parse_on_date_time(text)
                if when:
                    user_state["when"] = when
                    # Apply LLM timeIs if present
                    if nlp_entities.get("timeIs"):
                        user_state["when"]["timeIs"] = nlp_entities["timeIs"]

        # 2) If user said just "plan a journey" (or similar) without from/to,
        #    start fresh so we ask "Where from?" instead of reusing old state.
        #    BUT if this very message already gave us a from/to via LLM (e.g.
        #    "plan a journey from Neasden"), keep that slot and do NOT reset.
        if self._is_new_plan_without_locations(text) and "fromQuery" not in user_state and "toQuery" not in user_state:
            for key in (
                "fromQuery", "toQuery",
                "fromLocationId", "toLocationId",
                "fromLocationName", "toLocationName",
                "fromOptions", "toOptions",
                "fromQuestion", "toQuestion",
                "askedWhen", "when",
                "journey_planning_active",
                "awaiting_fromQuery", "awaiting_toQuery",
            ):
                user_state.pop(key, None)

        return self._continue_planning(text, user_state, now, username=username)

    def handle_structured_journey(
        self,
        from_text: str,
        to_text: str,
        from_id: str | None,
        to_id: str | None,
        date_str: str | None,
        time_str: str | None,
        now: datetime,
        username: str | None = None,
    ) -> Dict[str, Any]:
        """
        Entry point for structured inputs coming from the dedicated From/To/When
        text boxes. This bypasses the natural-language parsing and uses the
        provided values directly.
        """
        user_state = self.state["global"]
        # Reset journey-related state for a new structured request.
        for key in (
            "fromQuery",
            "toQuery",
            "fromLocationId",
            "toLocationId",
            "fromOptions",
            "toOptions",
            "fromQuestion",
            "toQuestion",
            "askedWhen",
            "when",
        ):
            user_state.pop(key, None)

        user_state["fromQuery"] = from_text.strip()
        user_state["toQuery"] = to_text.strip()

        # Build a when dict from explicit date/time if provided; otherwise fall back to "now".
        when = None
        if date_str and time_str:
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                when = {"datetime": dt, "timeIs": "Departing"}
            except ValueError:
                when = None
        if when is None:
            when = {"datetime": now, "timeIs": "Departing"}

        user_state["when"] = when

        # If we have concrete IDs from the autocomplete selection, use them
        # directly and bypass conversational disambiguation.
        if from_id and to_id:
            journeys = self.tfl_client.get_journeys(
                from_id=from_id,
                to_id=to_id,
                when=user_state["when"],
                via=user_state.get("_nlp_via"),
                mode=user_state.get("_nlp_mode"),
                journey_preference=user_state.get("_nlp_journey_preference"),
            )

            if not journeys:
                reply = "I couldn't find any journeys for that time. Try a different time?"
            else:
                reply = f"Here are your journey options from {from_text} to {to_text}"
                if username:
                    reply += f", {username}"
                reply += "."
            if self.tfl_client.last_url:
                reply += f"\n\n[debug] TfL URL: {self.tfl_client.last_url}"

            # Capture state snapshot for the response, then reset for the next turn.
            state_for_response = copy.deepcopy(user_state)
            out = {"reply": reply, "journeys": journeys, "state": state_for_response}
            if journeys and self.tfl_client.last_url:
                out["tfl_journey_url"] = self.tfl_client.last_url

            # After replying (whether or not any journeys were found), clear
            # journey-planning state so the next message starts from a clean slate.
            self._reset_journey_state(user_state)
            return out

        # Otherwise, delegate to the common planning flow, which will use from/to/when directly.
        return self._continue_planning(
            text=f"Plan a journey from {from_text} to {to_text}",
            user_state=user_state,
            now=now,
            username=username,
        )

    def _continue_planning(
        self,
        text: str,
        user_state: Dict[str, Any],
        now: datetime,
        username: str | None = None,
    ) -> Dict[str, Any]:
        """
        Shared core planning flow used by both free-text and structured entry points.
        """
        # Mark that we are in an active journey-planning conversation so the
        # /chat router keeps sending follow-up messages here until we either
        # reset the state or finish successfully.
        user_state["journey_planning_active"] = True

        # Extract LLM-derived TfL API parameters from user_state
        _via = user_state.get("_nlp_via")
        _mode = user_state.get("_nlp_mode")
        _journey_pref = user_state.get("_nlp_journey_preference")

        def _journey_reply(with_name: bool = False) -> str:
            from_name = user_state.get("fromLocationName") or user_state.get("fromQuery", "")
            to_name = user_state.get("toLocationName") or user_state.get("toQuery", "")
            base = f"Here are your journey options from {from_name} to {to_name}"
            if with_name and username:
                base += f", {username}"
            notes = []
            if _mode:
                notes.append(f"mode: {_mode}")
            if _via:
                notes.append(f"via {_via}")
            if _journey_pref:
                _JP_LABELS = {
                    "leastinterchange": "fewest changes",
                    "leasttime": "fastest",
                    "leastwalking": "least walking",
                }
                notes.append(_JP_LABELS.get(_journey_pref.lower(), _journey_pref))
            if notes:
                base += f" ({', '.join(notes)})"
            return base + "."

        # Override timeIs from LLM if available (e.g. user said "arrive by 6pm")
        if "_nlp_timeIs" in user_state and "when" in user_state:
            user_state["when"]["timeIs"] = user_state["_nlp_timeIs"]

        # 1) Resolve from and to through proper disambiguation.
        #
        # Previously this step sent raw place names directly to TfL's
        # Journey API (get_journeys_by_queries). If TfL could resolve the
        # names, it returned journeys and disambiguation was skipped
        # entirely. This caused silent wrong-location selection for
        # ambiguous queries like "Lavender Avenue" (exists in multiple
        # parts of London) — TfL would auto-pick one without presenting
        # alternatives.
        #
        # Now we always resolve from/to through the disambiguation engine
        # first. For unambiguous queries the engine auto-resolves instantly
        # (single candidate or high-confidence match); for ambiguous ones
        # the user gets to choose from well-labelled options.

        # 2) Ensure fromLocationId is resolved.
        if "fromLocationId" not in user_state:
            return self._handle_location_disambiguation(
                text, user_state, key_prefix="from", now=now, username=username
            )

        # 3) Ensure toLocationId is resolved.
        if "toLocationId" not in user_state:
            return self._handle_location_disambiguation(
                text, user_state, key_prefix="to", now=now, username=username
            )

        # 4) Both locations resolved — show draggable pin confirmation map before
        #    fetching the TfL journey, unless the user has already confirmed.
        if not user_state.get("_pins_confirmed"):
            return self._build_confirm_pins_response(user_state)

        # 5) If no when yet, assume NOW (we don't ask for time; extract or default).
        if "when" not in user_state:
            user_state["when"] = {"datetime": now, "timeIs": "Departing"}

        # 6) We have everything, fetch journeys with resolved IDs.
        journeys = self.tfl_client.get_journeys(
            from_id=user_state["fromLocationId"],
            to_id=user_state["toLocationId"],
            when=user_state["when"],
            via=_via,
            mode=_mode,
            journey_preference=_journey_pref,
        )

        if not journeys:
            reply = "I couldn't find any journeys for that time. Try a different time?"
        else:
            reply = _journey_reply(True)
            if self.tfl_client.last_url:
                reply += f"\n\n[debug] TfL URL: {self.tfl_client.last_url}"

        state_for_response = copy.deepcopy(user_state)
        out = {"reply": reply, "journeys": journeys, "state": state_for_response}
        if journeys and self.tfl_client.last_url:
            out["tfl_journey_url"] = self.tfl_client.last_url

        # After replying (whether or not any journeys were found), clear
        # journey-planning state so the next message starts from a clean slate.
        self._reset_journey_state(user_state)
        return out

    # ---- Internal helpers -----------------------------------------------

    # _apply_full_plan_if_present() has been removed.
    # Journey origin/destination extraction is now handled exclusively by the
    # LLM-based extractor (llm_entity_extractor.py). The LLM-extracted slots
    # (nlp_origin, nlp_destination) are passed directly to handle_message().

    def _is_new_plan_without_locations(self, text: str) -> bool:
        """
        True if the user is starting a new plan (e.g. "plan a journey") but did not
        give from/to in this message. We use this to reset state so we ask "Where from?"
        instead of reusing old journey state and possibly showing "no journeys".
        """
        if not (text or "").strip():
            return False
        t = text.lower().strip()
        has_plan = "plan" in t and ("journey" in t or "trip" in t or "route" in t)
        has_from_to = " from " in t and " to " in t
        return bool(has_plan and not has_from_to)

    # _maybe_extract_initial_intent() has been removed.
    # Partial-slot extraction (origin-only or destination-only) is now handled
    # by the LLM extractor. The LLM extracts whichever slots are present in the
    # user’s message and they are passed via nlp_origin / nlp_destination.

    def _handle_location_disambiguation(
        self,
        text: str,
        user_state: Dict[str, Any],
        key_prefix: str,
        now: datetime | None = None,
        username: str | None = None,
    ) -> Dict[str, Any]:
        """
        Handles journey location disambiguation with weighted scoring.

        Uses the journey disambiguation pipeline which scores candidates on:
          - proximity to user (browser geolocation)
          - proximity to the other end (destination if resolving origin, vice versa)
          - context clues ("near Kingsbury" anchor)
          - name similarity
          - user history (localStorage-persisted selection counts)
          - route feasibility (placeholder)

        Three possible outcomes:
          A. auto_select  — proceed silently with top candidate
          B. clarify      — show top candidates, ask user to pick
          C. show_list    — show ranked list (same cap)

        """
        query_key = f"{key_prefix}Query"
        chosen_id_key = f"{key_prefix}LocationId"
        chosen_name_key = f"{key_prefix}LocationName"
        pending_options_key = f"{key_prefix}Options"
        pending_question_key = f"{key_prefix}Question"

        def _continue_after_resolve() -> Dict[str, Any]:
            """Continue planning in the same request so we can show journey options."""
            if now is not None:
                return self._continue_planning(text, user_state, now, username=username)
            return {
                "reply": "Okay.",
                "journeys": [],
                "state": user_state,
            }

        # 1) If we don't have the raw text query yet: either ask for it, or use this message as the answer.
        if query_key not in user_state:
            user_state["journey_planning_active"] = True
            other_query_key = "toQuery" if key_prefix == "from" else "fromQuery"
            awaiting_key = f"awaiting_{key_prefix}Query"
            was_awaiting = bool(user_state.get(awaiting_key))
            if key_prefix == "to" and not was_awaiting and user_state.get(other_query_key) == (text or "").strip():
                user_state[awaiting_key] = True
                return {
                    "reply": f"Where are you travelling { 'from' if key_prefix == 'from' else 'to' }?",
                    "journeys": [],
                    "state": user_state,
                }
            if key_prefix == "to" and not was_awaiting and (text or "").strip().isdigit():
                user_state[awaiting_key] = True
                return {
                    "reply": "Where are you travelling to?",
                    "journeys": [],
                    "state": user_state,
                }
            if user_state.get(other_query_key) and not was_awaiting:
                direction = "from" if key_prefix == "from" else "to"
                user_state[awaiting_key] = True
                return {
                    "reply": f"Where are you travelling {direction}?",
                    "journeys": [],
                    "state": user_state,
                }
            user_state.pop(awaiting_key, None)
            cleaned = (text or "").strip()
            if cleaned and not self._is_new_plan_without_locations(cleaned):
                t = cleaned.lower()
                looks_like_full_request = (
                    (" go to " in t)
                    or (" get to " in t)
                    or (" take me to " in t)
                    or (" directions to " in t)
                    or (" navigate to " in t)
                    or (" from " in t and " to " in t)
                    or (" want to travel " in t)
                    or (" need to travel " in t)
                )
                if looks_like_full_request and not was_awaiting:
                    direction = "from" if key_prefix == "from" else "to"
                    user_state[awaiting_key] = True
                    return {
                        "reply": f"Where are you travelling {direction}? (Just the place name, e.g. 'Neasden Station')",
                        "journeys": [],
                        "state": user_state,
                    }
                if looks_like_full_request and was_awaiting:
                    # User pasted the full journey again instead of a single place — recover slots.
                    try:
                        slots = get_journey_slot_extractor().extract_journey_slots(cleaned)
                    except Exception:
                        slots = {}
                    slot = (
                        slots.get("origin")
                        if key_prefix == "from"
                        else slots.get("destination")
                    )
                    if slot and str(slot).strip():
                        user_state[query_key] = str(slot).strip()
                        user_state.pop(awaiting_key, None)
                    else:
                        direction = "starting point" if key_prefix == "from" else "destination"
                        return {
                            "reply": (
                                f"I still need just the {direction} (e.g. a street or station name), "
                                "not the full \"plan a journey from … to …\" sentence."
                            ),
                            "journeys": [],
                            "state": user_state,
                        }
                else:
                    user_state[query_key] = cleaned
            else:
                return {
                    "reply": f"Where are you travelling { 'from' if key_prefix == 'from' else 'to' }?",
                    "journeys": [],
                    "state": user_state,
                }

        # 2) If we already showed options and are waiting for a short answer, try to match it.
        if pending_options_key in user_state:
            options = user_state[pending_options_key]
            choice = text.strip().lower()

            # Match by number (e.g. "1", "2")
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(options):
                    selected = options[idx - 1]
                    user_state[chosen_id_key] = selected["id"]
                    user_state[chosen_name_key] = selected.get("name", "")
                    user_state.pop(pending_options_key, None)
                    user_state.pop(pending_question_key, None)
                    # Record selection for user history
                    self._record_location_choice(user_state, selected["id"])
                    return _continue_after_resolve()

            # Try exact or partial match on name or qualifier.
            matched = [
                o
                for o in options
                if choice in o["name"].lower()
                or (o.get("qualifier") and choice in o["qualifier"].lower())
            ]

            if len(matched) == 1:
                user_state[chosen_id_key] = matched[0]["id"]
                user_state[chosen_name_key] = matched[0].get("name", "")
                user_state.pop(pending_options_key, None)
                user_state.pop(pending_question_key, None)
                self._record_location_choice(user_state, matched[0]["id"])
                return _continue_after_resolve()

            if len(matched) > 1:
                return {
                    "reply": "Could you be a bit more specific? Tap or click a numbered label on the map, or type the option number (e.g. 1 or 2) or the full place name in the chat.",
                    "journeys": [],
                    "state": user_state,
                }

        # 4) Resolve candidates from Google Places + OSM, then coalesce same-name/nearby dupes.
        query = user_state[query_key]

        # Safety-net: strip any residual "near X" / "(near X)" from query before search.
        # The LLM should already have separated these, but if the user typed it directly
        # (e.g. as a rephrase) the query might still contain the context clause.
        query_clean, fallback_near = _extract_near_area(query)
        if fallback_near:
            query = query_clean
            user_state[query_key] = query_clean
            # Store the near_area if we don't already have one for this side
            side_name = "origin" if key_prefix == "from" else "destination"
            near_key = f"_nlp_{side_name}_near_area"
            if not user_state.get(near_key):
                user_state[near_key] = fallback_near
                print(f"[JourneyChatbot] Extracted residual near_area from query: {fallback_near!r}")

        # Pass near_area to the search so OSM/Google return geographically
        # biased candidates from the start (e.g. "Harrow Road, Sudbury, London"
        # rather than all Harrow Roads across London).
        search_near = user_state.get(f"_nlp_{'origin' if key_prefix == 'from' else 'destination'}_near_area") or user_state.get("_nlp_near_area")
        options = _fetch_merged_journey_location_options(query, max_results=20, near_area=search_near)

        if not options:
            user_state["journey_planning_active"] = True
            user_state["pending_location_rephrase_for"] = key_prefix
            side_word = "starting point" if key_prefix == "from" else "destination"
            reply = (
                f"I couldn't find anything matching '{query}' for your {side_word}. "
                "Reply with a single place or station name (e.g. 'Wembley Central' or 'Stonebridge Park'). "
                "I'll use your next message as that location."
            )
            if self.tfl_client.last_url:
                reply += f"\n\n[debug] TfL URL: {self.tfl_client.last_url}"
            return {
                "reply": reply,
                "journeys": [],
                "state": user_state,
            }

        # Build disambiguation context
        role = "origin" if key_prefix == "from" else "destination"

        # User location from browser geolocation (passed via user_state)
        user_lat = user_state.get("_user_lat")
        user_lon = user_state.get("_user_lon")

        # Other end coordinates (if already resolved)
        other_end_lat, other_end_lon = None, None
        if key_prefix == "from" and "toLocationId" in user_state:
            other_end_lat, other_end_lon = _parse_lat_lon_from_id(user_state["toLocationId"])
        elif key_prefix == "to" and "fromLocationId" in user_state:
            other_end_lat, other_end_lon = _parse_lat_lon_from_id(user_state["fromLocationId"])

        # Near area anchor — per-location context from LLM extraction.
        # e.g. "from Harrow Road (near Sudbury) to Lavender Avenue (near Wembley)"
        # When resolving origin, use origin_near_area; for destination, use destination_near_area.
        # Falls back to the legacy single near_area if per-location fields are absent.
        near_area_anchor = None
        if key_prefix == "from":
            near_area = user_state.get("_nlp_origin_near_area") or user_state.get("_nlp_near_area")
        else:
            near_area = user_state.get("_nlp_destination_near_area") or user_state.get("_nlp_near_area")
        near_area_structured = None
        if near_area:
            grounder = get_grounder()
            near_result = grounder.ground(near_area)
            if near_result:
                near_area_anchor = SpatialAnchor(
                    lat=near_result["lat"],
                    lng=near_result["lng"],
                    source="near_area",
                )
                # Extract structured hard-boundary fields from grounded result
                near_area_structured = NearAreaStructured(
                    suburb=near_result.get("suburb", "") or "",
                    borough=near_result.get("borough", "") or "",
                    postcode_prefix=near_result.get("postcode_prefix", "") or "",
                )
                print(
                    f"[JourneyChatbot] Near area anchor for {key_prefix}: {near_area} → "
                    f"({near_area_anchor.lat}, {near_area_anchor.lng}) | "
                    f"suburb={near_area_structured.suburb!r} "
                    f"borough={near_area_structured.borough!r} "
                    f"postcode_prefix={near_area_structured.postcode_prefix!r}"
                )

        # User history from localStorage (passed via user_state)
        user_history = JourneyUserHistory.from_dict(
            user_state.get("_user_journey_history")
        )

        # Convert options to DisambiguationCandidate list
        candidates = candidates_from_journey_options(options)

        # Build context
        ctx = JourneyDisambiguationContext(
            role=role,
            user_lat=user_lat,
            user_lon=user_lon,
            other_end_lat=other_end_lat,
            other_end_lon=other_end_lon,
            near_area_anchor=near_area_anchor,
            near_area_text=near_area if near_area else None,
            near_area_structured=near_area_structured,
            user_history=user_history,
        )

        # Run the disambiguation pipeline
        result = journey_disambiguate(query, candidates, ctx)

        # Drop low name-match candidates (full ranked list is still in `candidates`)
        filtered_ranked = [c for c in candidates if _journey_name_match_passes(c)]
        hidden_name_match_count = len(candidates) - len(filtered_ranked)

        # --- Handle outcome ---
        direction = "from" if key_prefix == "from" else "to"

        # A. Auto-select: proceed silently (only if name match clears the bar)
        if result.action == "auto_select" and result.chosen:
            if _journey_name_match_passes(result.chosen):
                user_state[chosen_id_key] = result.chosen.id
                user_state[chosen_name_key] = result.chosen.name
                self._record_location_choice(user_state, result.chosen.id)
                print(
                    f"[JourneyChatbot] Auto-selected '{result.chosen.name}' "
                    f"(score={result.top_score:.4f}, gap={result.confidence_gap:.4f})"
                )
                return _continue_after_resolve()
            print(
                f"[JourneyChatbot] Auto-select suppressed: name_similarity "
                f"{getattr(result.chosen, 'name_similarity', None)!r} "
                f"< {_JOURNEY_NAME_MATCH_MIN_DISPLAY:.0%}"
            )

        # Nothing left after name filter — ask user to rephrase
        if not filtered_ranked:
            user_state["journey_planning_active"] = True
            user_state["pending_location_rephrase_for"] = key_prefix
            side_word = "starting point" if key_prefix == "from" else "destination"
            pct = int(round(_JOURNEY_NAME_MATCH_MIN_DISPLAY * 100))
            reply = (
                f"I found possible places for '{query}', but none matched your wording closely enough "
                f"(name match below {pct}%). Try a clearer name, spelling, or add the area or postcode "
                f"for your {side_word}. Reply with that location and I'll use your next message."
            )
            if self.tfl_client.last_url:
                reply += f"\n\n[debug] TfL URL: {self.tfl_client.last_url}"
            return {
                "reply": reply,
                "journeys": [],
                "state": user_state,
            }

        # Only one viable candidate after filtering — resolve without prompting
        if len(filtered_ranked) == 1:
            only = filtered_ranked[0]
            user_state[chosen_id_key] = only.id
            user_state[chosen_name_key] = only.name
            self._record_location_choice(user_state, only.id)
            print(
                f"[JourneyChatbot] Auto-selected sole candidate after name filter: "
                f"'{only.name}' (name_similarity={getattr(only, 'name_similarity', None)!r}, "
                f"hidden_by_name={hidden_name_match_count})"
            )
            return _continue_after_resolve()

        # B/C. Clarify or show_list: cap options; list is from name-filtered ranking
        presented = filtered_ranked[:_JOURNEY_DISAMBIG_UI_MAX]

        # Convert candidates back to option dicts for storage
        presented_options = []
        for c in presented:
            opt = c.to_dict()
            # Ensure qualifier is preserved from original options
            for orig in options:
                if orig["id"] == c.id:
                    opt["qualifier"] = orig.get("qualifier", "")
                    opt["shortLabel"] = orig.get("shortLabel", c.name)
                    break
            presented_options.append(opt)

        user_state[pending_options_key] = presented_options

        # Build the reply message
        lines = []
        for i, o in enumerate(presented_options, 1):
            label = (o.get("shortLabel") or o.get("name") or "").strip()
            qual = (o.get("qualifier") or "").strip()
            score_val = o.get("score", 0)
            name_sim = o.get("name_similarity")
            if qual and "," in qual:
                qualifier = f" ({qual})"
            else:
                qualifier = ""
            score_str = f" — score: {score_val:.0%}" if score_val > 0 else ""
            if name_sim is not None:
                try:
                    score_str += f", name match: {float(name_sim):.0%}"
                except (TypeError, ValueError):
                    pass
            lines.append(f"  {i}. {label}{qualifier}{score_str}")

        if result.action == "clarify":
            intro = f"Did you mean one of these for '{query}' ({direction})?"
        else:
            intro = f"Which place did you mean for '{query}' ({direction})?"

        hidden_note = ""
        if hidden_name_match_count > 0:
            pct = int(round(_JOURNEY_NAME_MATCH_MIN_DISPLAY * 100))
            h = hidden_name_match_count
            hidden_note = (
                f"\n\n({h} other candidate{'s were' if h != 1 else ' was'} not shown because "
                f"{'their' if h != 1 else 'its'} name match to your query was below {pct}%.)"
            )

        footer = "\n\nTap or click a numbered label on the map, or type the number (e.g. 1 or 2) or the place name in the chat."

        question = intro + hidden_note + "\n" + "\n".join(lines) + footer
        user_state[pending_question_key] = question

        # Build place_disambiguation for frontend map
        place_disambiguation = _build_place_disambiguation_for_map(presented_options, query, direction)
        place_disambiguation["scores"] = {
            o.get("id", ""): o.get("score", 0) for o in presented_options
        }
        place_disambiguation["hidden_candidates_count"] = hidden_name_match_count
        place_disambiguation["name_match_min_display"] = _JOURNEY_NAME_MATCH_MIN_DISPLAY

        reply = question
        if self.tfl_client.last_url:
            reply += f"\n\n[debug] TfL URL: {self.tfl_client.last_url}"
        return {
            "reply": reply,
            "journeys": [],
            "state": user_state,
            "disambiguation": True,
            "place_disambiguation": place_disambiguation,
        }

    def _record_location_choice(self, user_state: Dict[str, Any], location_id: str) -> None:
        """Record a location selection for user history tracking.

        Stores in user_state so it can be sent back to the frontend
        for localStorage persistence.
        """
        history = user_state.get("_user_journey_history", {})
        chosen = history.get("chosen_locations", {})
        chosen[location_id] = chosen.get(location_id, 0) + 1
        history["chosen_locations"] = chosen
        user_state["_user_journey_history"] = history
        user_state["_updated_journey_history"] = True  # flag for frontend

    def _handle_datetime(
        self, text: str, user_state: Dict[str, Any], now: datetime
    ) -> Dict[str, Any]:
        """
        Ask for date+time in one question, and interpret short answers like 'now'.
        """
        if "askedWhen" not in user_state:
            user_state["askedWhen"] = True
            return {
                "reply": "When are you travelling? You can say things like 'now', 'tomorrow 9am', or 'arrive by 6pm'.",
                "journeys": [],
                "state": user_state,
            }

        parsed = parse_datetime_text(text, now=now)
        if not parsed:
            return {
                "reply": "I didn't quite catch the time. You can say 'now', 'tomorrow 9am', or 'arrive by 6pm'.",
                "journeys": [],
                "state": user_state,
            }

        user_state["when"] = parsed
        return {
            "reply": "Great, I'll plan around that time.",
            "journeys": [],
            "state": user_state,
        }

    def _build_confirm_pins_response(self, user_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a 'confirm_pins' response payload that tells the frontend to render
        a Google Maps with two draggable markers — one for origin, one for destination.
        The user can drag them to fine-tune the exact pickup/dropoff points before the
        route is fetched from TfL.
        """
        from_id = user_state.get("fromLocationId", "")
        to_id = user_state.get("toLocationId", "")
        from_lat, from_lon = _parse_lat_lon_from_id(from_id)
        to_lat, to_lon = _parse_lat_lon_from_id(to_id)

        from_name = user_state.get("fromLocationName") or user_state.get("fromQuery", "Origin")
        to_name = user_state.get("toLocationName") or user_state.get("toQuery", "Destination")

        return {
            "reply": (
                f"I've pinpointed **{from_name}** and **{to_name}** on the map. "
                "Drag either pin to fine-tune the exact spot, then tap **Plan Route** when you're ready."
            ),
            "journeys": [],
            "confirm_pins": {
                "from": {
                    "lat": from_lat,
                    "lon": from_lon,
                    "name": from_name,
                },
                "to": {
                    "lat": to_lat,
                    "lon": to_lon,
                    "name": to_name,
                },
            },
            "state": user_state,
        }

    def _reset_journey_state(self, user_state: Dict[str, Any]) -> None:
        """
        Reset journey-planning specific state after a journey has been planned
        and displayed, so that subsequent messages start from a clean slate.
        """
        for key in (
            "fromQuery",
            "toQuery",
            "fromLocationId",
            "toLocationId",
            "fromLocationName",
            "toLocationName",
            "fromOptions",
            "toOptions",
            "fromQuestion",
            "toQuestion",
            "askedWhen",
            "when",
            "journey_planning_active",
            "awaiting_fromQuery",
            "awaiting_toQuery",
            # LLM-extracted TfL API parameters
            "_nlp_via",
            "_nlp_mode",
            "_nlp_journey_preference",
            "_nlp_timeIs",
            "_nlp_near_area",
            "_nlp_origin_near_area",
            "_nlp_destination_near_area",
            "pending_location_rephrase_for",
            "_pins_confirmed",
            # User location (re-sent each request)
            "_user_lat",
            "_user_lon",
        ):
            user_state.pop(key, None)

