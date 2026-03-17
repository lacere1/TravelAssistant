"""
FSM-based Dialog State Tracker.

Replaces the flat-dictionary approach with an explicit finite state machine,
keyed per user. Each user has their own independent conversation state.

States:
  IDLE              - No active task, awaiting new user query
  PROCESSING        - Handling a direct query (timetable, disruption, traffic)
  AWAITING_DISAMBIGUATION - Presented options, waiting for user to pick one
  AWAITING_DISRUPTION_LINE - Asked user to specify a train line or bus route
  JOURNEY_COLLECTING_FROM - Journey planner: waiting for origin
  JOURNEY_COLLECTING_TO   - Journey planner: waiting for destination
  JOURNEY_DISAMBIGUATING  - Journey planner: disambiguating a location
  JOURNEY_READY           - Journey planner: all info collected, ready to plan

Transitions are explicit — each state defines what inputs move to which next state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List
from datetime import datetime


class DialogState(Enum):
    """All possible conversation states."""
    IDLE = auto()
    PROCESSING = auto()
    AWAITING_DISAMBIGUATION = auto()
    AWAITING_DISRUPTION_LINE = auto()
    JOURNEY_COLLECTING_FROM = auto()
    JOURNEY_COLLECTING_TO = auto()
    JOURNEY_DISAMBIGUATING = auto()
    JOURNEY_READY = auto()


@dataclass
class DisambiguationContext:
    """Context when the bot is waiting for the user to pick from options."""
    options: List[Dict[str, Any]]
    query: str
    mode: str  # 'bus', 'train', or 'timetable'
    # For journey planner: 'from' or 'to'
    direction: Optional[str] = None


@dataclass
class JourneyContext:
    """Context for an active journey planning conversation."""
    from_query: Optional[str] = None
    to_query: Optional[str] = None
    from_id: Optional[str] = None
    to_id: Optional[str] = None
    when: Optional[Dict[str, Any]] = None
    # Disambiguation state for journey locations
    from_options: Optional[List[Dict[str, Any]]] = None
    to_options: Optional[List[Dict[str, Any]]] = None
    disambiguating: Optional[str] = None  # 'from' or 'to'


@dataclass
class UserState:
    """
    Complete conversation state for a single user.

    This replaces the flat dictionary approach. Each field is typed and
    the current state is an explicit enum value.
    """
    state: DialogState = DialogState.IDLE
    # Last intent and entities from NLP
    last_intent: Optional[str] = None
    last_entities: Dict[str, Any] = field(default_factory=dict)
    last_confidence: float = 0.0
    # Disambiguation context (bus stop direction, station selection, etc.)
    disambiguation: Optional[DisambiguationContext] = None
    # What type of disruption line we're waiting for: 'train' or 'bus'
    awaiting_disruption_type: Optional[str] = None
    # Journey planner context
    journey: Optional[JourneyContext] = None
    # Per-user preferences
    last_chosen_stop_id: Optional[str] = None
    frequent_stops: Dict[str, int] = field(default_factory=dict)
    # Timestamp of last interaction
    last_interaction: Optional[datetime] = None

    def reset(self) -> None:
        """Reset to idle state, clearing all transient context."""
        self.state = DialogState.IDLE
        self.disambiguation = None
        self.awaiting_disruption_type = None
        self.journey = None
        # Clear last recognised intent/entities so a "new chat" truly starts fresh.
        self.last_intent = None
        self.last_entities.clear()
        self.last_confidence = 0.0
        # Note: We preserve preferences and history so features like
        # frequent_stops can continue to improve over time.

    def reset_journey(self) -> None:
        """Reset only journey-related state."""
        self.journey = None
        if self.state in (
            DialogState.JOURNEY_COLLECTING_FROM,
            DialogState.JOURNEY_COLLECTING_TO,
            DialogState.JOURNEY_DISAMBIGUATING,
            DialogState.JOURNEY_READY,
        ):
            self.state = DialogState.IDLE

    @property
    def is_in_journey_flow(self) -> bool:
        """Check if the user is currently in a journey planning flow."""
        return self.state in (
            DialogState.JOURNEY_COLLECTING_FROM,
            DialogState.JOURNEY_COLLECTING_TO,
            DialogState.JOURNEY_DISAMBIGUATING,
            DialogState.JOURNEY_READY,
        )

    @property
    def is_awaiting_input(self) -> bool:
        """Check if the bot is waiting for a specific user response."""
        return self.state in (
            DialogState.AWAITING_DISAMBIGUATION,
            DialogState.AWAITING_DISRUPTION_LINE,
            DialogState.JOURNEY_COLLECTING_FROM,
            DialogState.JOURNEY_COLLECTING_TO,
            DialogState.JOURNEY_DISAMBIGUATING,
        )


class DialogStateTracker:
    """
    Manages per-user dialog state using explicit FSM transitions.

    Usage:
        tracker = DialogStateTracker()
        state = tracker.get_state("user_123")
        state.state = DialogState.AWAITING_DISAMBIGUATION
        state.disambiguation = DisambiguationContext(...)
    """

    def __init__(self, session_timeout_minutes: int = 30):
        self._user_states: Dict[str, UserState] = {}
        self._session_timeout_minutes = session_timeout_minutes

    def get_state(self, user_key: str) -> UserState:
        """
        Get or create the state for a user.

        If the user's last interaction was more than session_timeout_minutes ago,
        resets their state (preserving preferences).
        """
        if user_key not in self._user_states:
            self._user_states[user_key] = UserState()

        user_state = self._user_states[user_key]

        # Check for session timeout
        if (
            user_state.last_interaction is not None
            and self._session_timeout_minutes > 0
        ):
            elapsed = (datetime.utcnow() - user_state.last_interaction).total_seconds()
            if elapsed > self._session_timeout_minutes * 60:
                # Preserve preferences but reset conversation context
                prefs = (user_state.last_chosen_stop_id, user_state.frequent_stops.copy())
                user_state.reset()
                user_state.last_chosen_stop_id = prefs[0]
                user_state.frequent_stops = prefs[1]

        user_state.last_interaction = datetime.utcnow()
        return user_state

    def has_state(self, user_key: str) -> bool:
        return user_key in self._user_states

    def reset_state(self, user_key: str) -> None:
        """Reset a user's state to IDLE."""
        if user_key in self._user_states:
            self._user_states[user_key].reset()

    # ---- Transition helpers ----
    # These encode the valid state transitions. The chatbot calls these
    # instead of directly setting state values, which prevents invalid transitions.

    def start_disambiguation(
        self, user_key: str, options: List[Dict[str, Any]], query: str, mode: str
    ) -> UserState:
        """Transition to AWAITING_DISAMBIGUATION state."""
        state = self.get_state(user_key)
        state.state = DialogState.AWAITING_DISAMBIGUATION
        state.disambiguation = DisambiguationContext(
            options=options, query=query, mode=mode
        )
        return state

    def resolve_disambiguation(self, user_key: str, chosen_id: str) -> UserState:
        """User picked a disambiguation option. Transition back to PROCESSING or IDLE."""
        state = self.get_state(user_key)
        # Record preference
        state.last_chosen_stop_id = chosen_id
        state.frequent_stops[chosen_id] = state.frequent_stops.get(chosen_id, 0) + 1
        state.disambiguation = None
        state.state = DialogState.PROCESSING
        return state

    def start_awaiting_disruption_line(
        self, user_key: str, disruption_type: str
    ) -> UserState:
        """
        Transition to AWAITING_DISRUPTION_LINE.
        disruption_type: 'train' or 'bus'
        """
        state = self.get_state(user_key)
        state.state = DialogState.AWAITING_DISRUPTION_LINE
        state.awaiting_disruption_type = disruption_type
        return state

    def resolve_disruption_line(self, user_key: str) -> UserState:
        """User provided a disruption line. Transition back to PROCESSING."""
        state = self.get_state(user_key)
        state.awaiting_disruption_type = None
        state.state = DialogState.PROCESSING
        return state

    def start_journey(self, user_key: str) -> UserState:
        """Begin a new journey planning flow."""
        state = self.get_state(user_key)
        state.journey = JourneyContext()
        state.state = DialogState.JOURNEY_COLLECTING_FROM
        return state

    def set_journey_from(
        self, user_key: str, from_query: str, from_id: Optional[str] = None
    ) -> UserState:
        """Set the journey origin and move to collecting destination."""
        state = self.get_state(user_key)
        if state.journey is None:
            state.journey = JourneyContext()
        state.journey.from_query = from_query
        state.journey.from_id = from_id
        state.state = DialogState.JOURNEY_COLLECTING_TO
        return state

    def set_journey_to(
        self, user_key: str, to_query: str, to_id: Optional[str] = None
    ) -> UserState:
        """Set the journey destination and move to READY."""
        state = self.get_state(user_key)
        if state.journey is None:
            state.journey = JourneyContext()
        state.journey.to_query = to_query
        state.journey.to_id = to_id
        state.state = DialogState.JOURNEY_READY
        return state

    def start_journey_disambiguation(
        self, user_key: str, direction: str, options: List[Dict[str, Any]], query: str
    ) -> UserState:
        """
        Journey planner needs disambiguation for a location.
        direction: 'from' or 'to'
        """
        state = self.get_state(user_key)
        state.state = DialogState.JOURNEY_DISAMBIGUATING
        if state.journey is None:
            state.journey = JourneyContext()
        state.journey.disambiguating = direction
        if direction == "from":
            state.journey.from_options = options
        else:
            state.journey.to_options = options
        state.disambiguation = DisambiguationContext(
            options=options, query=query, mode="journey", direction=direction
        )
        return state

    def resolve_journey_disambiguation(
        self, user_key: str, chosen_id: str
    ) -> UserState:
        """User picked a journey disambiguation option."""
        state = self.get_state(user_key)
        if state.journey is None:
            state.journey = JourneyContext()

        direction = state.journey.disambiguating
        if direction == "from":
            state.journey.from_id = chosen_id
            state.journey.from_options = None
            # If we have to_query, we're ready; otherwise collect to
            if state.journey.to_query:
                state.state = DialogState.JOURNEY_READY
            else:
                state.state = DialogState.JOURNEY_COLLECTING_TO
        else:
            state.journey.to_id = chosen_id
            state.journey.to_options = None
            state.state = DialogState.JOURNEY_READY

        state.journey.disambiguating = None
        state.disambiguation = None
        return state

    def finish_processing(self, user_key: str) -> UserState:
        """Processing complete, return to IDLE."""
        state = self.get_state(user_key)
        state.state = DialogState.IDLE
        state.disambiguation = None
        return state

    def to_dict(self, user_key: str) -> Dict[str, Any]:
        """
        Export user state as a dict for API responses / debugging.
        This provides backward compatibility with existing code that expects dict state.
        """
        state = self.get_state(user_key)
        result: Dict[str, Any] = {
            "dialog_state": state.state.name,
            "last_intent": state.last_intent,
            "last_confidence": state.last_confidence,
        }

        if state.disambiguation:
            result["disambiguation"] = {
                "options": state.disambiguation.options,
                "query": state.disambiguation.query,
                "mode": state.disambiguation.mode,
            }

        if state.awaiting_disruption_type:
            result["awaiting_disruption_line"] = state.awaiting_disruption_type

        if state.journey:
            j = state.journey
            result["journey"] = {
                "fromQuery": j.from_query,
                "toQuery": j.to_query,
                "fromLocationId": j.from_id,
                "toLocationId": j.to_id,
                "when": j.when,
            }

        return result

    def from_legacy_dict(self, user_key: str, legacy: Dict[str, Any]) -> UserState:
        """
        Import state from the old flat-dictionary format for backward compatibility.
        This allows gradual migration from the old chatbot.py state format.
        """
        state = self.get_state(user_key)

        # Journey planner state
        if legacy.get("fromQuery") or legacy.get("toQuery"):
            if state.journey is None:
                state.journey = JourneyContext()
            state.journey.from_query = legacy.get("fromQuery")
            state.journey.to_query = legacy.get("toQuery")
            state.journey.from_id = legacy.get("fromLocationId")
            state.journey.to_id = legacy.get("toLocationId")
            state.journey.when = legacy.get("when")

            if legacy.get("fromOptions"):
                state.state = DialogState.JOURNEY_DISAMBIGUATING
                state.journey.disambiguating = "from"
                state.journey.from_options = legacy["fromOptions"]
            elif legacy.get("toOptions"):
                state.state = DialogState.JOURNEY_DISAMBIGUATING
                state.journey.disambiguating = "to"
                state.journey.to_options = legacy["toOptions"]
            elif state.journey.from_query and state.journey.to_query:
                state.state = DialogState.JOURNEY_READY
            elif state.journey.from_query:
                state.state = DialogState.JOURNEY_COLLECTING_TO
            else:
                state.state = DialogState.JOURNEY_COLLECTING_FROM

        # Timetable disambiguation
        if legacy.get("timetable_disambiguation"):
            dis = legacy["timetable_disambiguation"]
            state.state = DialogState.AWAITING_DISAMBIGUATION
            state.disambiguation = DisambiguationContext(
                options=dis.get("options", []),
                query=dis.get("query", ""),
                mode=dis.get("timetable_mode", "timetable"),
            )

        # Disruption line awaiting
        if legacy.get("awaiting_disruption_line"):
            state.state = DialogState.AWAITING_DISRUPTION_LINE
            state.awaiting_disruption_type = legacy["awaiting_disruption_line"]

        return state

    def to_legacy_dict(self, user_key: str) -> Dict[str, Any]:
        """
        Export state as the old flat-dictionary format for backward compatibility
        with the existing chatbot.py and journey_planner.py.
        """
        state = self.get_state(user_key)
        legacy: Dict[str, Any] = {
            "origin": None,
            "destination": None,
            "timetable_disambiguation": None,
            "awaiting_disruption_line": state.awaiting_disruption_type,
        }

        if state.disambiguation and state.state == DialogState.AWAITING_DISAMBIGUATION:
            legacy["timetable_disambiguation"] = {
                "options": state.disambiguation.options,
                "query": state.disambiguation.query,
                "timetable_mode": state.disambiguation.mode,
            }

        if state.journey:
            j = state.journey
            legacy["fromQuery"] = j.from_query
            legacy["toQuery"] = j.to_query
            legacy["fromLocationId"] = j.from_id
            legacy["toLocationId"] = j.to_id
            legacy["when"] = j.when
            legacy["fromOptions"] = j.from_options
            legacy["toOptions"] = j.to_options
            legacy["journey_planning_active"] = state.is_in_journey_flow

        return legacy
