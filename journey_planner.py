from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class JourneyLeg:
    mode: str
    detail: str
    duration: int


class TflJourneyClient:
    """
    Minimal client used by the journey planner.
    """

    def search_places(self, query: str) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        base = [
            {"id": "Oxford Circus", "name": "Oxford Circus", "shortLabel": "Oxford Circus", "qualifier": "Underground"},
            {"id": "Harlesden", "name": "Harlesden", "shortLabel": "Harlesden", "qualifier": "Rail / Underground"},
            {"id": "Neasden", "name": "Neasden", "shortLabel": "Neasden", "qualifier": "Underground"},
        ]
        q_lower = q.lower()
        return [p for p in base if q_lower in p["name"].lower()] or base

    def get_journeys_by_queries(
        self,
        from_query: str,
        to_query: str,
        when: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not from_query or not to_query:
            return []
        dt: datetime = when.get("datetime") or datetime.utcnow()
        duration = 35
        legs = [
            {
                "mode": "Walk",
                "detail": f"Walk to {from_query} stop",
                "duration": 5,
            },
            {
                "mode": "Bus",
                "detail": f"Bus towards {to_query}",
                "duration": 20,
            },
            {
                "mode": "Walk",
                "detail": f"Walk from {to_query} stop to destination",
                "duration": 10,
            },
        ]
        arrival = dt.replace(minute=(dt.minute + duration) % 60)
        return [
            {
                "departure": dt.strftime("%H:%M"),
                "arrival": arrival.strftime("%H:%M"),
                "duration": duration,
                "legs": legs,
            }
        ]


class JourneyChatbot:
    """
    Small journey planner that returns simple London journeys based on free‑text or structured input.
    """

    def __init__(self) -> None:
        self.client = TflJourneyClient()

    def _default_when(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        return {"datetime": now or datetime.utcnow(), "timeIs": "Departing"}

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
        when = self._default_when(now)
        if date_str and time_str:
            try:
                when["datetime"] = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            except ValueError:
                when = self._default_when(now)

        journeys = self.client.get_journeys_by_queries(from_text, to_text, when)
        if not journeys:
            reply = "I couldn't find a simple journey for that request."
        else:
            reply = "Here are your journey options across the London network."

        return {
            "reply": reply,
            "journeys": journeys,
            "state": {},
        }

    def handle_message(
        self,
        text: str,
        now: datetime,
        username: str | None = None,
    ) -> Dict[str, Any]:
        lowered = (text or "").lower()
        if " from " in lowered and " to " in lowered:
            try:
                _, after_from = lowered.split(" from ", 1)
                from_part, to_part = after_from.split(" to ", 1)
                from_text = from_part.strip().title()
                to_text = to_part.strip().title()
                when = self._default_when(now)
                journeys = self.client.get_journeys_by_queries(from_text, to_text, when)
                if not journeys:
                    reply = f"I couldn't find a simple journey from {from_text} to {to_text}."
                else:
                    reply = f"Here is a journey from {from_text} to {to_text} on the London network."
                return {"reply": reply, "journeys": journeys, "state": {}}
            except ValueError:
                pass

        return {
            "reply": "Tell me something like “Plan a journey from Harlesden to Oxford Circus”.",
            "journeys": [],
            "state": {},
        }


