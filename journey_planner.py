"""
Journey planner backend integrated into the main app.

This file brings over the TfL journey-planning chatbot, its lightweight
datetime NLP, and the TfL Journey API wrapper from the separate
`seperateAdd` project so they can be used inside the current app.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import os
import re
import copy
import requests

try:
    # Microsoft Recognizers-Text for robust date/time parsing
    from recognizers_text import Culture  # type: ignore[import]
    from recognizers_date_time import DateTimeRecognizer  # type: ignore[import]

    _MS_RECOGNIZERS_AVAILABLE = True
except Exception as e:
    # Log the real import error so it's visible in the console.
    print(f"[journey_planner] Recognizers-Text import failed: {e}")
    _MS_RECOGNIZERS_AVAILABLE = False
    Culture = None  # type: ignore[assignment]
    DateTimeRecognizer = None  # type: ignore[assignment]


# ---- Lightweight + Microsoft Recognizers datetime NLP -----------------------


@dataclass
class ParsedWhen:
    date: datetime
    timeIs: str  # "Departing" or "Arriving"


_datetime_model = None


def _get_datetime_model():
    """
    Lazily initialize and cache the Microsoft Recognizers-Text datetime model.
    Falls back gracefully if the package is missing or fails to load.
    """
    global _datetime_model
    if not _MS_RECOGNIZERS_AVAILABLE:
        # Debug: recognizers-text not installed or failed import
        print("[journey_planner] Microsoft Recognizers-Text not available; using legacy datetime parser.")
        return None
    if _datetime_model is not None:
        return _datetime_model
    try:
        recognizer = DateTimeRecognizer(Culture.English)
        _datetime_model = recognizer.get_datetime_model()
        print("[journey_planner] Microsoft Recognizers-Text datetime model initialised successfully.")
    except Exception as e:
        print(f"[journey_planner] Failed to initialise Recognizers-Text datetime model: {e}")
        _datetime_model = None
    return _datetime_model


def parse_datetime_text(text: str, now: datetime) -> Optional[dict]:
    """
    Parse natural language date/time phrases like:
      - 'now'
      - 'tomorrow 9am'
      - 'today 18:30'
      - 'arrive by 6pm'
      - 'next Monday at 7:30'

    Uses Microsoft Recognizers-Text when available for rich parsing,
    and falls back to the original lightweight pattern-based logic.

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

    # Try Microsoft Recognizers-Text first
    model = _get_datetime_model()
    if model is not None:
        try:
            results = model.parse(raw)
            if results:
                # Prefer datetime / time results
                best = None
                for r in results:
                    ttype = (r.type_name or "").lower()
                    if "datetime" in ttype or "time" in ttype or "date" in ttype:
                        best = r
                        break
                if best is None:
                    best = results[0]

                res = best.resolution or {}
                values = res.get("values") or []
                if values:
                    v = values[0]
                    # v may contain 'value' (datetime or date) and 'time' / 'start' etc.
                    dt_str = v.get("value") or v.get("start") or v.get("time")
                    if dt_str:
                        dt: Optional[datetime] = None
                        try:
                            # recognizers-text uses ISO-like formats; fromisoformat handles most of them
                            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                        except Exception:
                            # If only a time is present (no date), combine with today
                            m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", dt_str)
                            if m:
                                hour = int(m.group(1))
                                minute = int(m.group(2))
                                dt = datetime(
                                    year=now.year,
                                    month=now.month,
                                    day=now.day,
                                    hour=hour,
                                    minute=minute,
                                )
                        if dt is not None:
                            # Decide whether this is a departure or arrival time
                            time_is = "Departing"
                            # If the original text clearly says "arrive by", treat it as arrival
                            if "arrive by" in t or "arriving by" in t:
                                time_is = "Arriving"
                            print(f"[journey_planner] Parsed datetime with Recognizers-Text: {dt.isoformat()} ({time_is}) from {raw!r}")
                            return {"datetime": dt, "timeIs": time_is}
        except Exception as e:
            # Fall through to the legacy parser
            print(f"[journey_planner] Recognizers-Text parsing error, falling back to legacy parser: {e}")

    # Legacy lightweight parser (original behaviour) as fallback
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
GOOGLE_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


def _parse_lat_lon_from_id(place_id: str) -> tuple[float | None, float | None]:
    """If place_id is 'lat,lng' (e.g. from Google Places), return (lat, lon); else (None, None)."""
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


# Greater London bounds (SW and NE corners) – same as frontend autocomplete.
# Restricts Places API results to London only.
LONDON_BOUNDS_SW = {"latitude": 51.28, "longitude": -0.489}
LONDON_BOUNDS_NE = {"latitude": 51.686, "longitude": 0.236}


def _google_places_search(query: str, api_key: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Call Google Places API (new) searchText to find places matching the query.
    Restricted to Greater London via locationRestriction. Returns options in the same
    shape as TfL disambiguation: { "id": "lat,lng", "name", "qualifier", "shortLabel" }.
    TfL journey API accepts "lat,lng" as from_id/to_id.
    """
    if not query or not api_key:
        return []
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
    }
    body = {
        "textQuery": query,
        "locationRestriction": {
            "rectangle": {
                "low": LONDON_BOUNDS_SW,
                "high": LONDON_BOUNDS_NE,
            }
        },
    }
    try:
        resp = requests.post(
            GOOGLE_PLACES_SEARCH_URL,
            json=body,
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    places = data.get("places") or []
    results: List[Dict[str, Any]] = []
    for p in places[:max_results]:
        loc = p.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None:
            continue
        place_id = f"{lat},{lon}"
        display = p.get("displayName") or {}
        name = display.get("text", query) if isinstance(display, dict) else str(display)
        addr = p.get("formattedAddress") or ""
        results.append(
            {
                "id": place_id,
                "name": name,
                "qualifier": addr,
                "shortLabel": name,
            }
        )
    return results


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

    def get_journeys(self, from_id: str, to_id: str, when: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetches journeys between two resolved place IDs around a given datetime.
        TfL format: date=YYYYMMDD, time=HHmm (no separators), timeIs=Departing|Arriving.
        """
        dt: datetime = when["datetime"]
        time_is: str = when["timeIs"]  # "Departing" or "Arriving"

        url = f"{TFL_BASE_URL}/journey/journeyresults/{from_id}/to/{to_id}"
        self.last_url = url

        # Primary call: TfL format – date=YYYYMMDD, time=HHmm (no separators), timeIs=Departing|Arriving.
        params = self._auth_params()
        params.update(
            {
                "date": dt.strftime("%Y%m%d"),
                "time": dt.strftime("%H%M"),
                "timeIs": time_is,
            }
        )
        try:
            resp = self.session.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            journeys = data.get("journeys") or []
            if journeys:
                return _journeys_to_summaries(journeys)
        except Exception:
            pass

        # Fallback: try date/time again with hyphen/colon in case of strict parsing elsewhere.
        try:
            params_alt = self._auth_params()
            params_alt.update(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%H:%M"),
                    "timeIs": time_is,
                }
            )
            resp2 = self.session.get(url, params=params_alt, timeout=8)
            resp2.raise_for_status()
            data2 = resp2.json()
            journeys2 = data2.get("journeys") or []
            if journeys2:
                return _journeys_to_summaries(journeys2)
        except Exception:
            pass

        # Fallback: if still no journeys, try without date/time so TfL can suggest alternatives.
        try:
            params_fallback = self._auth_params()
            resp2 = self.session.get(url, params=params_fallback, timeout=8)
            resp2.raise_for_status()
            data2 = resp2.json()
            journeys2 = data2.get("journeys") or []
            return _journeys_to_summaries(journeys2)
        except Exception:
            return []

    def get_journeys_by_queries(
        self, from_query: str, to_query: str, when: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Tries the journey API with from/to as place names (e.g. "Royal Borough of Kingston upon Thames").
        If TfL returns itineraries directly, returns them; otherwise returns [] so the caller can disambiguate.
        """
        if not from_query.strip() or not to_query.strip():
            return []
        dt: datetime = when["datetime"]
        time_is: str = when["timeIs"]
        url = f"{TFL_BASE_URL}/journey/journeyresults/{requests.utils.quote(from_query.strip())}/to/{requests.utils.quote(to_query.strip())}"
        self.last_url = url

        # Primary call: TfL format – date=YYYYMMDD, time=HHmm, timeIs=Departing|Arriving.
        params = self._auth_params()
        params.update(
            {
                "date": dt.strftime("%Y%m%d"),
                "time": dt.strftime("%H%M"),
                "timeIs": time_is,
            }
        )
        try:
            resp = self.session.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            journeys = data.get("journeys") or []
            if journeys:
                return _journeys_to_summaries(journeys)
        except Exception:
            pass

        # Fallback: date/time with hyphen and colon.
        try:
            params_alt = self._auth_params()
            params_alt.update(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%H:%M"),
                    "timeIs": time_is,
                }
            )
            resp2 = self.session.get(url, params=params_alt, timeout=8)
            resp2.raise_for_status()
            data2 = resp2.json()
            journeys2 = data2.get("journeys") or []
            if journeys2:
                return _journeys_to_summaries(journeys2)
        except Exception:
            pass

        # Fallback without date/time – let TfL pick nearby services.
        try:
            params_fallback = self._auth_params()
            resp3 = self.session.get(url, params=params_fallback, timeout=8)
            resp3.raise_for_status()
            data3 = resp3.json()
            journeys3 = data3.get("journeys") or []
            return _journeys_to_summaries(journeys3)
        except Exception:
            return []

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
    Uses Google Places API for place disambiguation when GOOGLE_PLACES_API_KEY is set.
    """

    def __init__(self):
        self.tfl_client = TflJourneyClient()
        self.google_places_api_key = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()
        self.state: Dict[str, Dict[str, Any]] = {"global": {}}

    def reset(self) -> None:
        """Reset all journey-planning state for a fresh conversation."""
        self.state["global"].clear()

    # ---- Public API -----------------------------------------------------

    def handle_message(
        self,
        text: str,
        now: datetime,
        username: str | None = None,
        nlp_origin: str | None = None,
        nlp_destination: str | None = None,
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

        Returns a JSON structure the frontend can render:
        {
          "reply": "...",
          "journeys": [...],     # optional, structured results
          "state": {...}         # debug/inspection if desired
        }
        """
        user_state = self.state["global"]

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
            # Continue planning immediately with the updated query.
            return self._continue_planning(text, user_state, now, username=username)

        # 1) Text-based extraction: "from X to Y" / "to Y from X" patterns.
        self._apply_full_plan_if_present(text, user_state, now=now)
        if "fromQuery" not in user_state or "toQuery" not in user_state:
            self._maybe_extract_initial_intent(text, user_state)

        # 2) NLP fallback: use the NER-extracted slots for any slot still missing.
        #    This makes use of the SpaCy journey slot extractor that runs in
        #    ner_processor.py before the message reaches the journey planner.
        if nlp_origin and "fromQuery" not in user_state:
            user_state["fromQuery"] = nlp_origin
            print(f"[JourneyChatbot] fromQuery from NLP: {nlp_origin!r}")
        if nlp_destination and "toQuery" not in user_state:
            user_state["toQuery"] = nlp_destination
            print(f"[JourneyChatbot] toQuery from NLP: {nlp_destination!r}")

        # 3) If user said just "plan a journey" (or similar) without from/to,
        #    start fresh so we ask "Where from?" instead of reusing old state.
        #    BUT if this very message already gave us a from/to (e.g.
        #    "plan a journey from Neasden"), keep that slot and do NOT reset.
        if self._is_new_plan_without_locations(text) and "fromQuery" not in user_state and "toQuery" not in user_state:
            for key in (
                "fromQuery", "toQuery",
                "fromLocationId", "toLocationId",
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
            )

            if not journeys:
                reply = "I couldn't find any journeys for that time. Try a different time?"
            else:
                reply = "Here are your journey options." + (f", {username}." if username else ".")
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
        def _journey_reply(with_name: bool = False) -> str:
            base = "Here are your journey options."
            return base + (f", {username}." if username and with_name else ".")

        # 1) If we have from, to and when, try the journey API with place names first.
        if "fromQuery" in user_state and "toQuery" in user_state and "when" in user_state:
            journeys = self.tfl_client.get_journeys_by_queries(
                from_query=user_state["fromQuery"],
                to_query=user_state["toQuery"],
                when=user_state["when"],
            )
            if journeys:
                reply = _journey_reply(True)
                if self.tfl_client.last_url:
                    reply += f"\n\n[debug] TfL URL: {self.tfl_client.last_url}"
                state_for_response = copy.deepcopy(user_state)
                out = {"reply": reply, "journeys": journeys, "state": state_for_response}
                if self.tfl_client.last_url:
                    out["tfl_journey_url"] = self.tfl_client.last_url

                # After a journey has been planned and displayed, clear journey-planning state.
                self._reset_journey_state(user_state)
                return out

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

        # 4) If no when yet, assume NOW (we don't ask for time; extract or default).
        if "when" not in user_state:
            user_state["when"] = {"datetime": now, "timeIs": "Departing"}

        # 5) We have everything, fetch journeys with resolved IDs.
        journeys = self.tfl_client.get_journeys(
            from_id=user_state["fromLocationId"],
            to_id=user_state["toLocationId"],
            when=user_state["when"],
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

    def _apply_full_plan_if_present(self, text: str, user_state: Dict[str, Any], now: datetime) -> None:
        """
        If the message contains both origin and destination, extract them and
        set fromQuery / toQuery.  Handles both word orders:
          - "from X to Y"   (standard)
          - "to Y from X"   (reversed, e.g. "take me to Oxford Circus from Neasden")

        Uses original casing for place names passed to the TfL API.
        Strips trailing date/time phrases from each location span.
        """
        t = text.strip()

        from_part: Optional[str] = None
        to_part: Optional[str] = None

        # ---- Try "from X to Y" (standard order) ----
        m = re.search(
            r'\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+(?:at|on)\b.*)?$',
            t, re.IGNORECASE,
        )
        if m:
            from_part = _strip_datetime_suffix(m.group(1).strip())
            to_part   = _strip_datetime_suffix(m.group(2).strip())

        # ---- Try "to Y from X" (reversed order) ----
        # Greedy .* at the front makes the engine find the LAST "to" before "from".
        if not from_part or not to_part:
            m = re.search(
                r'.*\bto\s+(.+?)\s+from\s+(.+?)(?:\s+(?:at|on)\b.*)?$',
                t, re.IGNORECASE,
            )
            if m:
                # Example of this pattern we DO want:
                #   "get me to Oxford Circus from Neasden"
                # Example we do NOT want:
                #   "I want to plan a journey from Harlesden"
                candidate_to   = _strip_datetime_suffix(m.group(1).strip())
                candidate_from = _strip_datetime_suffix(m.group(2).strip())

                def _looks_like_non_place_control_phrase(s: str) -> bool:
                    """Heuristic: phrases like 'plan a journey', 'get me', 'need to' should not be treated as destinations."""
                    if not s:
                        return True
                    # Strip leading quotes / brackets so examples like
                    # `"plan a journey from Harlesden"` don't slip through.
                    s_lower = s.lower().lstrip(" '\"“”‘’(")
                    leading_verbs = (
                        "plan",
                        "planning",
                        "get",
                        "getting",
                        "need",
                        "needing",
                        "want",
                        "wanting",
                        "go",
                        "going",
                        "leave",
                        "leaving",
                        "start",
                        "starting",
                        "travel",
                        "travelling",
                        "traveling",
                    )
                    if any(s_lower.startswith(v + " ") for v in leading_verbs):
                        return True
                    # Phrases that clearly describe actions, not places
                    if "journey" in s_lower or "trip" in s_lower or "route" in s_lower:
                        return True
                    return False

                # Only accept the reversed pattern if the "to" side looks like a
                # genuine place name rather than a control phrase.
                if not _looks_like_non_place_control_phrase(candidate_to):
                    to_part   = candidate_to
                    from_part = candidate_from

        if not from_part or not to_part:
            return

        user_state["fromQuery"] = from_part
        user_state["toQuery"]   = to_part

        for key in (
            "fromLocationId", "toLocationId",
            "fromOptions",    "toOptions",
            "fromQuestion",   "toQuestion",
            "askedWhen",      "when",
        ):
            user_state.pop(key, None)

        when = _parse_journey_datetime(text, now=now) or _parse_on_date_time(text)
        user_state["when"] = when or {"datetime": now, "timeIs": "Departing"}

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

    def _maybe_extract_initial_intent(self, text: str, user_state: Dict[str, Any]):
        """
        Partial-slot extraction: sets whichever of fromQuery / toQuery is still
        missing after _apply_full_plan_if_present ran.

        Handles:
          - "from X" only  → sets fromQuery
          - "to X" only    → sets toQuery (greedily finds the LAST "to")
        Uses original casing; strips trailing date/time noise.
        """
        t = text.strip()

        def _starts_with_verb_or_pronoun(phrase: str) -> bool:
            """
            Heuristic: if a candidate place phrase starts with a verb or pronoun
            (e.g. 'plan a journey', 'i want', 'we need'), we do NOT treat it as
            a from/to location.
            """
            if not phrase:
                return False
            first = phrase.strip().split()[0].lower().strip(" '\"“”‘’(),.")
            bad_starts = {
                # Pronouns / determiners
                "i", "i'm", "im", "me", "you", "we", "they", "he", "she", "it",
                "my", "your", "our", "their", "his", "her", "its",
                "this", "that", "these", "those",
                # Common journey verbs
                "go", "going", "get", "getting", "take", "taking",
                "plan", "planning", "travel", "travelling", "traveling",
                "leave", "leaving", "depart", "departing",
                "start", "starting", "head", "heading", "navigate",
                "navigating", "walk", "walking", "drive", "driving",
                "catch", "catching", "need", "needing", "want", "wanting",
                "know", "see", "make", "do", "be", "have",
            }
            return first in bad_starts

        if "fromQuery" not in user_state:
            # "from X" – stop before "to", "at", "on"
            m = re.search(
                r'\bfrom\s+(.+?)(?:\s+(?:to|at|on)\b.*)?$',
                t, re.IGNORECASE,
            )
            if m:
                part = _strip_datetime_suffix(m.group(1).strip())
                if part and not _starts_with_verb_or_pronoun(part):
                    user_state.setdefault("fromQuery", part)

        if "toQuery" not in user_state:
            # Greedy .* finds the LAST "to" (avoids "I want TO travel TO X" picking "travel")
            m = re.search(
                r'.*\bto\s+(.+?)(?:\s+(?:from|at|on)\b.*)?$',
                t, re.IGNORECASE,
            )
            if m:
                part = _strip_datetime_suffix(m.group(1).strip())
                if part and not _starts_with_verb_or_pronoun(part):
                    user_state.setdefault("toQuery", part)

    def _handle_location_disambiguation(
        self,
        text: str,
        user_state: Dict[str, Any],
        key_prefix: str,
        now: datetime | None = None,
        username: str | None = None,
    ) -> Dict[str, Any]:
        """
        Handles both asking disambiguation questions and interpreting very short answers.
        key_prefix is 'from' or 'to'. When a location is resolved (single option or user pick),
        continues the flow in the same request so we can resolve the other location and
        return journey options without requiring another user message.
        """
        query_key = f"{key_prefix}Query"
        chosen_id_key = f"{key_prefix}LocationId"
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

        # 1) If we don't have the raw text query yet: either ask for it, or use this message as the answer (e.g. user replied "neasden station").
        if query_key not in user_state:
            user_state["journey_planning_active"] = True  # so next message is routed to journey planner
            # Don't use the same message for "to" that we already used for "from" (we may be in same request after _continue_after_resolve).
            other_query_key = "toQuery" if key_prefix == "from" else "fromQuery"
            if key_prefix == "to" and user_state.get(other_query_key) == (text or "").strip():
                return {
                    "reply": f"Where are you travelling { 'from' if key_prefix == 'from' else 'to' }?",
                    "journeys": [],
                    "state": user_state,
                }
            # Don't treat a disambiguation-style reply (e.g. "5") as a place name for "to" when we're continuing after resolving "from".
            if key_prefix == "to" and (text or "").strip().isdigit():
                return {
                    "reply": "Where are you travelling to?",
                    "journeys": [],
                    "state": user_state,
                }
            # If the other slot is already set (meaning this message already gave us one
            # location via slot extraction), don't treat the full message as a place name
            # for this slot — the extractor already had a chance and found nothing.
            # Ask the user explicitly instead (e.g. "get me to Neasden" → ask "Where from?").
            # BUT: if we already asked (awaiting_key flag), the user's message IS the answer.
            awaiting_key = f"awaiting_{key_prefix}Query"
            if user_state.get(other_query_key) and not user_state.get(awaiting_key):
                direction = "from" if key_prefix == "from" else "to"
                user_state[awaiting_key] = True  # mark that we've asked, next reply is the answer
                return {
                    "reply": f"Where are you travelling {direction}?",
                    "journeys": [],
                    "state": user_state,
                }
            # Clear the awaiting flag — we've received the answer.
            user_state.pop(awaiting_key, None)
            # If the user sent a place name in reply to our question, use it and continue.
            if (text or "").strip() and not self._is_new_plan_without_locations(text):
                user_state[query_key] = text.strip()
                # Fall through to step 3 to get options / disambiguate for this query.
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

            # Match by number (e.g. "1", "2") like bus stop disambiguation.
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(options):
                    user_state[chosen_id_key] = options[idx - 1]["id"]
                    user_state.pop(pending_options_key, None)
                    user_state.pop(pending_question_key, None)
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
                user_state.pop(pending_options_key, None)
                user_state.pop(pending_question_key, None)
                return _continue_after_resolve()

            if len(matched) > 1:
                return {
                    "reply": "Could you be a bit more specific? For example, reply with the number (e.g. 1 or 2) or the full place name.",
                    "journeys": [],
                    "state": user_state,
                }

            # no match, fall back to asking again with options

        # 3) Get place options: Google Places API if key set, else TfL disambiguation.
        query = user_state[query_key]
        if self.google_places_api_key:
            options = _google_places_search(query, self.google_places_api_key, max_results=5)
        else:
            options = self.tfl_client.disambiguate_location(query)

        if not options:
            # Remember which side ('from' or 'to') we need a rephrased location for,
            # so the next short reply (e.g. "neasden") can be treated as a new
            # location value for that side.
            user_state["journey_planning_active"] = True
            user_state["pending_location_rephrase_for"] = key_prefix
            reply = (
                f"I couldn't find anything matching '{query}'. "
                "Could you rephrase or give a nearby station or area?"
            )
            if self.tfl_client.last_url:
                reply += f"\n\n[debug] TfL URL: {self.tfl_client.last_url}"
            return {
                "reply": reply,
                "journeys": [],
                "state": user_state,
            }

        if len(options) == 1:
            user_state[chosen_id_key] = options[0]["id"]
            return _continue_after_resolve()

        # Multiple options: ask disambiguation (same style as bus stop disambiguation).
        user_state[pending_options_key] = options
        direction = "from" if key_prefix == "from" else "to"
        lines = [f"  {i}. {o['name']}" + (f" ({o['qualifier']})" if o.get("qualifier") else "") for i, o in enumerate(options, 1)]
        question = (
            f"Which place did you mean for '{query}' ({direction})?\n"
            + "\n".join(lines)
            + "\n\nReply with the number (e.g. 1 or 2) or the place name."
        )
        user_state[pending_question_key] = question

        # Build place_disambiguation for frontend map: options with lat, lon, label, name (id is often "lat,lng" from Google Places).
        place_disambiguation = _build_place_disambiguation_for_map(options, query, direction)

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
            "fromOptions",
            "toOptions",
            "fromQuestion",
            "toQuestion",
            "askedWhen",
            "when",
            "journey_planning_active",
            "awaiting_fromQuery",
            "awaiting_toQuery",
        ):
            user_state.pop(key, None)

