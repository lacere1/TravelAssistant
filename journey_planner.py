"""
Journey planner backend integrated into the main app.

This file brings over the TfL journey-planning chatbot and the TfL Journey API wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import os
import re
import requests


TFL_BASE_URL = "https://api.tfl.gov.uk"


@dataclass
class JourneyLeg:
    mode: str
    detail: str


@dataclass
class JourneySummary:
    departure: str
    arrival: str
    duration: int
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
        self.last_url: str | None = None

    def disambiguate_location(self, query: str) -> List[Dict[str, Any]]:
        """
        Calls the journey results endpoint with a free-text location to let TfL
        perform disambiguation. Returns a list of options with id/name/qualifier.
        """
        if not query:
            return []

        params = self._auth_params()
        url = f"{TFL_BASE_URL}/journey/journeyresults/{requests.utils.quote(query)}/to/bank"
        self.last_url = url

        try:
            resp = self.session.get(url, params=params, timeout=5)
            resp.raise_for_status()
        except Exception:
            return []

        data = resp.json()

        from_disamb = data.get("fromLocationDisambiguation") or {}
        disambiguation_options = from_disamb.get("disambiguationOptions") or []

        options: List[Dict[str, Any]] = []
        for option in disambiguation_options:
            place = option.get("place", {})
            parameter_value = option.get("parameterValue", "")
            common_name = place.get("commonName", query)

            if not parameter_value:
                continue

            options.append(
                {
                    "id": parameter_value,
                    "name": common_name,
                    "shortLabel": common_name,
                }
            )

        return options

    def get_journeys(self, from_id: str, to_id: str, when: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetches journeys between two resolved place IDs around a given datetime.
        """
        dt: datetime = when["datetime"]
        time_is: str = when.get("timeIs", "Departing")

        url = f"{TFL_BASE_URL}/journey/journeyresults/{from_id}/to/{to_id}"
        self.last_url = url

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

        try:
            params_fallback = self._auth_params()
            resp2 = self.session.get(url, params=params_fallback, timeout=8)
            resp2.raise_for_status()
            data2 = resp2.json()
            journeys2 = data2.get("journeys") or []
            return _journeys_to_summaries(journeys2)
        except Exception:
            return []

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

            detail_parts = []
            if instr:
                detail_parts.append(instr)

            legs_summary.append({
                "mode": mode or "Walk",
                "detail": " – ".join(detail_parts) if detail_parts else "",
            })

        if not start_dt or not end_dt or duration is None:
            continue

        summaries.append(
            {
                "departure": start_dt.strftime("%H:%M"),
                "arrival": end_dt.strftime("%H:%M"),
                "duration": duration,
                "legs": legs_summary,
            }
        )
    return summaries


def _parse_tfl_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


class JourneyChatbot:
    """
    Simple stateful dialogue manager for a TfL journey-planning assistant.
    """

    def __init__(self):
        self.tfl_client = TflJourneyClient()
        self.state: Dict[str, Dict[str, Any]] = {"global": {}}

    def handle_message(
        self, text: str, now: datetime, username: str | None = None
    ) -> Dict[str, Any]:
        """
        Core entry point called from Flask.

        Returns a JSON structure the frontend can render.
        """
        user_state = self.state["global"]

        self._apply_full_plan_if_present(text, user_state)
        if "fromQuery" not in user_state or "toQuery" not in user_state:
            self._maybe_extract_initial_intent(text, user_state)

        return self._continue_planning(text, user_state, now, username=username)

    def _continue_planning(
        self,
        text: str,
        user_state: Dict[str, Any],
        now: datetime,
        username: str | None = None,
    ) -> Dict[str, Any]:
        """
        Shared core planning flow.
        """
        def _journey_reply(with_name: bool = False) -> str:
            base = "Here are your journey options."
            return base + (f", {username}." if username and with_name else ".")

        if "fromLocationId" not in user_state:
            return self._handle_location_disambiguation(
                text, user_state, key_prefix="from"
            )

        if "toLocationId" not in user_state:
            return self._handle_location_disambiguation(
                text, user_state, key_prefix="to"
            )

        if "when" not in user_state:
            return self._handle_datetime(text, user_state, now)

        journeys = self.tfl_client.get_journeys(
            from_id=user_state["fromLocationId"],
            to_id=user_state["toLocationId"],
            when=user_state["when"],
        )

        if not journeys:
            reply = "I couldn't find any journeys for that time. Try a different time?"
        else:
            reply = _journey_reply(True)

        out = {"reply": reply, "journeys": journeys}
        if journeys and self.tfl_client.last_url:
            out["tfl_journey_url"] = self.tfl_client.last_url
        return out

    def _apply_full_plan_if_present(self, text: str, user_state: Dict[str, Any]) -> None:
        """
        If the message looks like "Plan a journey from X to Y", set fromQuery and toQuery.
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

        orig = text.strip()
        try:
            _, after_from_orig = orig.split(" from ", 1)
            from_orig, to_orig = after_from_orig.split(" to ", 1)
            user_state["fromQuery"] = from_orig.strip()
            user_state["toQuery"] = to_orig.strip()
        except ValueError:
            user_state["fromQuery"] = from_part
            user_state["toQuery"] = to_part

        for key in ("fromLocationId", "toLocationId", "fromOptions", "toOptions", "when"):
            user_state.pop(key, None)

    def _maybe_extract_initial_intent(self, text: str, user_state: Dict[str, Any]):
        """
        Look for patterns like 'from X to Y' in the initial sentence.
        """
        lowered = text.lower()
        if " from " in lowered and " to " in lowered:
            try:
                _, after_from = lowered.split(" from ", 1)
                from_part, to_part = after_from.split(" to ", 1)
                user_state.setdefault("fromQuery", from_part.strip())
                user_state.setdefault("toQuery", to_part.strip())
            except ValueError:
                pass

    def _handle_location_disambiguation(
        self, text: str, user_state: Dict[str, Any], key_prefix: str
    ) -> Dict[str, Any]:
        """
        Handles both asking disambiguation questions and interpreting answers.
        key_prefix is 'from' or 'to'.
        """
        query_key = f"{key_prefix}Query"
        chosen_id_key = f"{key_prefix}LocationId"
        pending_options_key = f"{key_prefix}Options"

        if query_key not in user_state:
            return {
                "reply": f"Where are you travelling {'from' if key_prefix == 'from' else 'to'}?",
                "journeys": [],
            }

        if pending_options_key in user_state:
            options = user_state[pending_options_key]
            choice = text.strip().lower()

            matched = [
                o
                for o in options
                if choice in o["name"].lower()
            ]

            if len(matched) == 1:
                user_state[chosen_id_key] = matched[0]["id"]
                user_state.pop(pending_options_key, None)
                return {
                    "reply": "Okay.",
                    "journeys": [],
                }

            if len(matched) > 1:
                return {
                    "reply": "Could you be more specific?",
                    "journeys": [],
                }

        query = user_state[query_key]
        options = self.tfl_client.disambiguate_location(query)

        if not options:
            return {
                "reply": f"I couldn't find anything matching '{query}'.",
                "journeys": [],
            }

        if len(options) == 1:
            user_state[chosen_id_key] = options[0]["id"]
            return {
                "reply": f"Got it: {options[0]['name']}.",
                "journeys": [],
            }

        user_state[pending_options_key] = options

        return {
            "reply": f"Which {query} did you mean?",
            "journeys": [],
            "disambiguation": True,
        }

    def _handle_datetime(
        self, text: str, user_state: Dict[str, Any], now: datetime
    ) -> Dict[str, Any]:
        """
        Ask for date+time.
        """
        if "askedWhen" not in user_state:
            user_state["askedWhen"] = True
            return {
                "reply": "When are you travelling?",
                "journeys": [],
            }

        user_state["when"] = {"datetime": now, "timeIs": "Departing"}
        return {
            "reply": "Great.",
            "journeys": [],
        }
