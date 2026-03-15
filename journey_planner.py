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
import requests


# ---- Lightweight datetime NLP ----------------------------------------------


@dataclass
class ParsedWhen:
    date: datetime
    timeIs: str  # "Departing" or "Arriving"


def parse_datetime_text(text: str, now: datetime) -> Optional[dict]:
    """
    Extremely lightweight parser for phrases like:
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
    t = text.strip().lower()

    if t == "now":
        return {"datetime": now, "timeIs": "Departing"}

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
        """
        dt: datetime = when["datetime"]
        time_is: str = when["timeIs"]  # "Departing" or "Arriving"

        url = f"{TFL_BASE_URL}/journey/journeyresults/{from_id}/to/{to_id}"
        self.last_url = url

        # Primary call: respect the requested date/time.
        params = self._auth_params()
        params.update(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M"),
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
            # Fall through to a looser retry below.
            pass

        # Fallback: if no journeys were found for the exact time, try again
        # without date/time so TfL can suggest reasonable alternatives.
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

        # Primary call: with explicit date/time.
        params = self._auth_params()
        params.update(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M"),
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

        # Fallback without date/time – let TfL pick nearby services.
        try:
            params_fallback = self._auth_params()
            resp2 = self.session.get(url, params=params_fallback, timeout=8)
            resp2.raise_for_status()
            data2 = resp2.json()
            journeys2 = data2.get("journeys") or []
            return _journeys_to_summaries(journeys2)
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
    """

    def __init__(self):
        self.tfl_client = TflJourneyClient()
        self.state: Dict[str, Dict[str, Any]] = {"global": {}}

    # ---- Public API -----------------------------------------------------

    def handle_message(
        self, text: str, now: datetime, username: str | None = None
    ) -> Dict[str, Any]:
        """
        Core entry point called from Flask.

        Returns a JSON structure the frontend can render:
        {
          "reply": "...",
          "journeys": [...],     # optional, structured results
          "state": {...}         # debug/inspection if desired
        }
        """
        user_state = self.state["global"]

        # 1) If the message looks like a full "from X to Y [on date at time]", use it and reset state.
        self._apply_full_plan_if_present(text, user_state)
        if "fromQuery" not in user_state or "toQuery" not in user_state:
            self._maybe_extract_initial_intent(text, user_state)

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

            out = {"reply": reply, "journeys": journeys}
            if journeys and self.tfl_client.last_url:
                out["tfl_journey_url"] = self.tfl_client.last_url
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
                out = {"reply": reply, "journeys": journeys}
                if self.tfl_client.last_url:
                    out["tfl_journey_url"] = self.tfl_client.last_url
                return out

        # 2) Ensure fromLocationId is resolved.
        if "fromLocationId" not in user_state:
            return self._handle_location_disambiguation(
                text, user_state, key_prefix="from"
            )

        # 3) Ensure toLocationId is resolved.
        if "toLocationId" not in user_state:
            return self._handle_location_disambiguation(
                text, user_state, key_prefix="to"
            )

        # 4) Ask for / interpret date & time.
        if "when" not in user_state:
            return self._handle_datetime(text, user_state, now)

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

        out = {"reply": reply, "journeys": journeys}
        if journeys and self.tfl_client.last_url:
            out["tfl_journey_url"] = self.tfl_client.last_url
        return out

    # ---- Internal helpers -----------------------------------------------

    def _apply_full_plan_if_present(self, text: str, user_state: Dict[str, Any]) -> None:
        """
        If the message looks like "Plan a journey from X to Y on 2026-02-16 at 18:00",
        set fromQuery and toQuery, clear resolution state, and set when if date/time present.
        """
        lowered = text.lower().strip()
        if " from " not in lowered or " to " not in lowered:
            return
        try:
            _, after_from = lowered.split(" from ", 1)
            from_part, to_part = after_from.split(" to ", 1)
        except ValueError:
            return
        from_part = from_part.strip()
        to_part = to_part.strip()
        # Strip trailing date/time from to_part
        to_part = re.sub(
            r"\s+on\s+\d{4}-\d{2}-\d{2}\s+at\s+\d{1,2}:\d{2}\s*$",
            "", to_part, flags=re.IGNORECASE,
        )
        to_part = re.sub(
            r"\s+at\s+\d{1,2}:\d{2}\s*$", "", to_part, flags=re.IGNORECASE
        )
        to_part = to_part.strip()
        user_state["fromQuery"] = from_part
        user_state["toQuery"] = to_part
        for key in (
            "fromLocationId", "toLocationId", "fromOptions", "toOptions",
            "fromQuestion", "toQuestion", "askedWhen", "when",
        ):
            user_state.pop(key, None)
        when = _parse_on_date_time(text)
        if when is not None:
            user_state["when"] = when

    def _maybe_extract_initial_intent(self, text: str, user_state: Dict[str, Any]):
        """Look for patterns like 'from X to Y' in the initial sentence."""
        lowered = text.lower()
        if " from " in lowered and " to " in lowered:
            try:
                _, after_from = lowered.split(" from ", 1)
                from_part, to_part = after_from.split(" to ", 1)
                to_part = re.sub(r"\s+on\s+\d{4}-\d{2}-\d{2}\s+at\s+\d{1,2}:\d{2}\s*$", "", to_part, flags=re.IGNORECASE)
                to_part = re.sub(r"\s+at\s+\d{1,2}:\d{2}\s*$", "", to_part, flags=re.IGNORECASE)
                user_state.setdefault("fromQuery", from_part.strip())
                user_state.setdefault("toQuery", to_part.strip())
            except ValueError:
                pass

    def _handle_location_disambiguation(
        self, text: str, user_state: Dict[str, Any], key_prefix: str
    ) -> Dict[str, Any]:
        """
        Handles both asking disambiguation questions and interpreting very short answers.
        key_prefix is 'from' or 'to'.
        """
        query_key = f"{key_prefix}Query"
        chosen_id_key = f"{key_prefix}LocationId"
        pending_options_key = f"{key_prefix}Options"
        pending_question_key = f"{key_prefix}Question"

        # 1) If we don't even have the raw text query yet, ask for it.
        if query_key not in user_state:
            return {
                "reply": f"Where are you travelling { 'from' if key_prefix == 'from' else 'to' }?",
                "journeys": [],
                "state": user_state,
            }

        # 2) If we already showed options and are waiting for a short answer, try to match it.
        if pending_options_key in user_state:
            options = user_state[pending_options_key]
            choice = text.strip().lower()

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
                # silently continue – next call will progress the flow
                reply = "Okay."
                if self.tfl_client.last_url:
                return {
                    "reply": reply,
                    "journeys": [],
                    "state": user_state,
                }

            if len(matched) > 1:
                # still ambiguous, one clarification only
                return {
                    "reply": "Could you be a bit more specific? For example, say the full station name.",
                    "journeys": [],
                    "state": user_state,
                }

            # no match, fall back to asking again with options

        # 3) Call TfL for this query and build a friendly question if needed.
        query = user_state[query_key]
        options = self.tfl_client.disambiguate_location(query)

        if not options:
            reply = (
                f"I couldn't find anything matching '{query}'. "
                "Could you rephrase or give a nearby station or area?"
            )
            if self.tfl_client.last_url:
            return {
                "reply": reply,
                "journeys": [],
                "state": user_state,
            }

        if len(options) == 1:
            user_state[chosen_id_key] = options[0]["id"]
            reply = f"Got it: {options[0]['name']}."
            if self.tfl_client.last_url:
            return {
                "reply": reply,
                "journeys": [],
                "state": user_state,
            }

        # Ask user to pick via fillable fields; remember options for later.
        user_state[pending_options_key] = options
        question = (
            f"Which {query} did you mean? Please use the fillable fields below."
        )
        user_state[pending_question_key] = question

        reply = question
        if self.tfl_client.last_url:
        return {
            "reply": reply,
            "journeys": [],
            "state": user_state,
            "disambiguation": True,
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

