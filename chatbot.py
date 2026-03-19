"""
Smart Traffic Query Assistant - Main Chatbot Core
Handles NLP processing, traffic data fetching, and ML predictions.

Updated to use FSM-based DialogStateTracker for per-user state management.
"""
from nlp_processor import NLPProcessor
from transport_api import TransportDataFetcher
from dialog_state import DialogStateTracker, DialogState, DisambiguationContext, UserState
from disambiguation_engine import (
    DisambiguationEngine,
    DisambiguationCandidate,
    SpatialAnchor,
    UserContext,
    candidates_from_tfl_matches,
    anchor_from_places,
    get_disambiguation_engine,
)
from places_grounder import get_grounder
import json
import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Any, List, Tuple

class TrafficChatbot:
    def __init__(self):
        """Initialize chatbot with NLP, API, and ML components"""
        self.nlp = NLPProcessor()
        self.transport_api = TransportDataFetcher()

        # ---- New: FSM-based per-user dialog state tracker ----
        # The DialogStateTracker manages per-user state with explicit FSM transitions.
        # For backward compatibility, self.conversation_state still works as before
        # but is now per-user (keyed by user_key in process_message).
        self.state_tracker = DialogStateTracker(session_timeout_minutes=30)

        # Per-user conversation state dicts (replaces the single global dict).
        # Each user gets their own dict, preventing concurrent-user state corruption.
        self._user_conversation_states: Dict[str, Dict[str, Any]] = {}

        # Default conversation state for backward compat (used when no user_key)
        self.conversation_state: Dict[str, Any] = {
            'origin': None,
            'destination': None,
            'timetable_disambiguation': None,
            'awaiting_disruption_line': None,
        }

        # Per-user preference/history for disambiguation
        self._user_preferences: Dict[str, Dict[str, Any]] = {}

        print("Traffic Chatbot initialized successfully")

    def reset_user(self, user_key: Optional[str] = None) -> None:
        """
        Reset all conversation and dialog state for a given user.

        Used by the UI "new chat" action so a conversation can start from a
        completely clean slate while still preserving long-term preferences
        tracked inside DialogStateTracker (e.g. frequent stops).
        """
        key = user_key or "_anon"

        # Reset FSM state (clears intents/entities and transient dialog flags).
        if self.state_tracker.has_state(key):
            self.state_tracker.reset_state(key)

        # Reset legacy per-user conversation dict to its initial shape.
        if key in self._user_conversation_states:
            self._user_conversation_states[key] = {
                'origin': None,
                'destination': None,
                'timetable_disambiguation': None,
                'awaiting_disruption_line': None,
            }

        # Clear per-user disambiguation preferences (last chosen stop, frequencies).
        if key in self._user_preferences:
            self._user_preferences.pop(key, None)

    def _get_user_conversation_state(self, user_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the conversation state dict for a specific user.
        Creates a fresh state if the user doesn't have one yet.
        This replaces the old global self.conversation_state with per-user state.
        """
        key = user_key or "_anon"
        if key not in self._user_conversation_states:
            self._user_conversation_states[key] = {
                'origin': None,
                'destination': None,
                'timetable_disambiguation': None,
                'awaiting_disruption_line': None,
            }
        return self._user_conversation_states[key]

    def _get_user_state(self, user_key: Optional[str] = None) -> UserState:
        """Get the FSM state for a user (for new code paths)."""
        key = user_key or "_anon"
        return self.state_tracker.get_state(key)

    def _sync_fsm_from_dict(self, user_key: Optional[str] = None) -> None:
        """Sync the FSM state from the per-user conversation dict."""
        key = user_key or "_anon"
        conv_state = self._get_user_conversation_state(key)
        self.state_tracker.from_legacy_dict(key, conv_state)

    def _sync_dict_from_fsm(self, user_key: Optional[str] = None) -> None:
        """Sync the per-user conversation dict from the FSM state."""
        key = user_key or "_anon"
        fsm_legacy = self.state_tracker.to_legacy_dict(key)
        conv_state = self._get_user_conversation_state(key)
        conv_state.update(fsm_legacy)
    
    def process_message(
        self,
        user_message: str,
        user_key: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process user message and generate response.

        Uses FSM-based DialogStateTracker for per-user state management.
        The state machine determines whether we're in the middle of a
        disambiguation flow, awaiting a disruption line, etc.

        Args:
            user_message: User's natural language query
            user_key: Optional per-user/session key for preferences (e.g. username or '_anon').
            username: Optional display name for personalised replies (e.g. greeting, goodbye).

        Returns:
            dict: Response with message, intent, entities, and confidence
        """
        # ---- Per-user state (replaces global self.conversation_state) ----
        # Use per-user conversation state dict instead of the old global dict.
        # Point self.conversation_state at this user's dict for the duration of
        # this call, so all the existing code that reads/writes
        # self.conversation_state works correctly per-user.
        self.conversation_state = self._get_user_conversation_state(user_key)

        # Also sync FSM state for new code paths
        user_fsm_state = self._get_user_state(user_key)

        # Dialogue State Tracking: if we're waiting for bus/train disambiguation, treat this as a reply
        disamb = self.conversation_state.get('timetable_disambiguation')
        if disamb and isinstance(disamb, dict):
            intent, selected, confidence = self._classify_disambiguation_reply(
                user_message.strip(), disamb.get('options', [])
            )

            if intent == 'cancel':
                mode_text = 'train' if disamb.get('timetable_mode') == 'train' else 'bus'
                self.conversation_state['timetable_disambiguation'] = None
                # FSM transition: cancel disambiguation, return to idle
                self.state_tracker.finish_processing(user_key or "_anon")
                response_message = {
                    'primary': "No problem.",
                    'details': None,
                    'alternatives': [],
                    'next_steps': f"Ask for {mode_text} times again whenever you like."
                }
                return {
                    'message': self._format_response(response_message, 'ask_timetable', {}),
                    'intent': 'ask_timetable',
                    'entities': {},
                    'confidence': 1.0,
                    'conversation_state': self.conversation_state.copy()
                }

            if intent == 'choose_option' and selected is not None and confidence >= 0.5:
                # User chose an option; either fetch timetable by stop (bus/train station) or filter by direction (train platform/direction)
                is_train_direction = disamb.get('train_direction_disambiguation') is True
                self.conversation_state['timetable_disambiguation'] = None
                # FSM transition: resolve disambiguation
                self.state_tracker.resolve_disambiguation(user_key or "_anon", selected.get('id', ''))
                timetable_mode = disamb.get('timetable_mode', 'bus')
                if is_train_direction:
                    # Filter existing timetable data by chosen direction; no re-fetch
                    stored_data = disamb.get('timetable_data') or {}
                    chosen_direction = selected.get('id') or selected.get('direction') or selected.get('name')
                    all_trains = stored_data.get('train_arrivals', [])
                    filtered_trains = [t for t in all_trains if (t.get('direction') or '') == chosen_direction]
                    filtered_by_direction = {chosen_direction: filtered_trains[:10]} if filtered_trains else {}
                    timetable_data = {
                        'stop_name': stored_data.get('stop_name', disamb.get('query', '')),
                        'train_arrivals': filtered_trains,
                        'train_arrivals_by_direction': filtered_by_direction,
                        'bus_arrivals': [],
                        'bus_arrivals_by_destination': {},
                        'bus_arrivals_grouped': {},
                    }
                else:
                    self._update_user_preference(user_key, selected['id'], selected.get('name'))
                    timetable_data = self.transport_api.get_tfl_timetable_by_stop_id(
                        selected['id'], mode_filter=timetable_mode, stop_name=selected.get('name')
                    )
                if timetable_data and isinstance(timetable_data, dict) and 'error' not in timetable_data:
                    response_message = self._format_timetable_response(timetable_data, timetable_mode)
                    response_message['timetable_data'] = timetable_data
                    formatted = self._format_response(response_message, 'ask_timetable', {})
                    return {
                        'message': formatted,
                        'intent': 'ask_timetable',
                        'entities': {},
                        'confidence': min(0.95, 0.7 + confidence * 0.25),
                        'conversation_state': self.conversation_state.copy(),
                        'timetable': timetable_data,
                    }
                if not is_train_direction and isinstance(timetable_data, dict) and timetable_data.get('error') == 'no_arrivals':
                    stop_name = timetable_data.get('stop_name', selected.get('name', ''))
                    self.conversation_state['timetable_disambiguation'] = None
                    response_message = {
                        'primary': f"No arrivals currently available for {stop_name}.",
                        'details': None,
                        'alternatives': [],
                        'next_steps': "Try again later or check another stop."
                    }
                    return {
                        'message': self._format_response(response_message, 'ask_timetable', {}),
                        'intent': 'ask_timetable',
                        'entities': {},
                        'confidence': 0.9,
                        'conversation_state': self.conversation_state.copy()
                    }
                # Fall through to normal processing if API failed (only for stop disambiguation, not train direction)
                if not is_train_direction:
                    self.conversation_state['timetable_disambiguation'] = disamb  # restore
            else:
                # Unclear or low-confidence reply: re-prompt with the same options
                prompt = self._build_disambiguation_prompt(disamb)
                primary = prompt['primary']
                if intent == 'unclear' or confidence < 0.5:
                    primary = "I didn't quite get that. " + primary
                response_message = {
                    'primary': primary,
                    'details': prompt.get('details'),
                    'alternatives': [],
                    'next_steps': prompt.get('next_steps')
                }
                formatted = self._format_response(response_message, 'ask_timetable', {})
                return {
                    'message': formatted,
                    'intent': 'ask_timetable',
                    'entities': {},
                    'confidence': confidence,
                    'conversation_state': self.conversation_state.copy()
                }
        
        # Follow-up: user replying with just a line/route after "couldn't find a train/bus"
        awaiting = self.conversation_state.get('awaiting_disruption_line')
        if awaiting and user_message and hasattr(self.nlp, 'parse_line_or_route_followup'):
            parsed = self.nlp.parse_line_or_route_followup(user_message)
            if parsed is not None:
                value, kind = parsed
                self.conversation_state['awaiting_disruption_line'] = None
                synthetic = f"bus status {value}" if kind == 'bus' else f"train status {value}"
                response_message = self._handle_transit_multimodal({}, synthetic)
                if isinstance(response_message, dict) and 'awaiting_line' in response_message:
                    response_message.pop('awaiting_line')
                formatted_response = self._format_response(response_message, 'ask_transit_disruption', {})
                self._update_conversation_state({}, 'ask_transit_disruption')
                return {
                    'message': formatted_response,
                    'intent': 'ask_transit_disruption',
                    'entities': {},
                    'confidence': 0.9,
                    'conversation_state': self.conversation_state.copy()
                }
        
        # Step 1: NLP Processing - Intent and Entity Extraction
        nlp_result = self.nlp.process(user_message)
        intent = nlp_result['intent']
        entities = nlp_result['entities']
        confidence = nlp_result['confidence']
        
        # Step 2: Update conversation state with new entities
        self._update_conversation_state(entities, intent)
        
        # Step 3: Generate response based on intent
        response_message = self._generate_response(
            intent, entities, user_message, user_key=user_key, username=username
        )
        
        # Step 4: Format response using template (ETA + route, alternatives, next steps)
        formatted_response = self._format_response(response_message, intent, entities)
        
        return {
            'message': formatted_response,
            'intent': intent,
            'entities': entities,
            'confidence': confidence,
            'conversation_state': self.conversation_state.copy(),
            'timetable': response_message.get('timetable_data'),
            'disruption': response_message.get('disruption'),
            # Expose current timetable disambiguation state (if any) so the frontend
            # can render rich UI like embedded maps for "Which direction for ...?"
            'timetable_disambiguation': self.conversation_state.get('timetable_disambiguation'),
        }
    
    def _update_conversation_state(self, entities: Dict[str, str], intent: str):
        """Update conversation state with extracted entities"""
        if entities.get('origin'):
            self.conversation_state['origin'] = entities['origin']
        if entities.get('destination'):
            self.conversation_state['destination'] = entities['destination']

    def _resolve_disambiguation_reply(self, user_message: str, options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Resolve user reply to a bus stop option (Dialogue State Tracking / Disambiguation Engine).
        Accepts any message that *contains* the identifying info: option number (1, first, one),
        platform (Stop AA, AA), or direction name (e.g. Willesden). Order: platform → towards → numeric.
        """
        if not user_message or not options:
            return None
        text = user_message.strip().lower()
        # Normalize whitespace for phrase matching
        text_norm = ' ' + re.sub(r'\s+', ' ', text) + ' '

        # 1) Match by platform/stop letter anywhere: "stop AA", "stop aa", "Stop BB", or word "aa"/"bb"
        for opt in options:
            platform = (opt.get('platform') or '').strip().upper()
            if not platform:
                continue
            pl = platform.lower()
            # "stop aa", "stop  aa", "stop aa please"
            if re.search(r'\bstop\s+' + re.escape(pl) + r'\b', text_norm):
                return opt
            # Platform as whole word (e.g. "aa", "bb") to avoid matching inside words
            if len(pl) >= 2 and re.search(r'\b' + re.escape(pl) + r'\b', text_norm):
                return opt
            # Single-letter platform: "stop a" for platform "A"
            if len(pl) == 1 and re.search(r'\bstop\s+' + re.escape(pl) + r'\b', text_norm):
                return opt

        # 1b) Match by platform number (train): "platform 1", "platform 2", "platform 3"
        for opt in options:
            platform = (opt.get('platform') or '').strip()
            if not platform or not platform.isdigit():
                continue
            if re.search(r'\bplatform\s*' + re.escape(platform) + r'\b', text_norm):
                return opt
            if re.search(r'\b' + re.escape(platform) + r'\b', text_norm) and re.search(r'\bplatform\b', text_norm):
                return opt

        # 2) Match by "towards" direction name (anywhere in message)
        for opt in options:
            towards = (opt.get('towards') or '').strip().lower()
            if not towards:
                continue
            if towards in text:
                return opt
            # Match significant words from towards, but require the word to be
            # distinctive (≥5 chars) AND not be a common location/query word.
            # This prevents "bank" matching as a towards direction when the user
            # is asking about "Bank station".
            _towards_ignore = {'stop', 'station', 'road', 'street', 'lane', 'avenue',
                               'park', 'hill', 'green', 'town', 'centre', 'gate', 'bridge'}
            for word in towards.split():
                if len(word) >= 5 and word not in _towards_ignore and re.search(r'\b' + re.escape(word) + r'\b', text_norm):
                    return opt

        # 2b) Match by train direction (northbound, southbound, eastbound, westbound, clockwise, anticlockwise)
        for opt in options:
            direction = (opt.get('direction') or opt.get('name') or opt.get('id') or '').strip().lower()
            if not direction:
                continue
            if direction in text:
                return opt
            # Allow "north" for Northbound, "south" for Southbound, etc.
            if direction == 'northbound' and re.search(r'\bnorth\b', text_norm):
                return opt
            if direction == 'southbound' and re.search(r'\bsouth\b', text_norm):
                return opt
            if direction == 'eastbound' and re.search(r'\beast\b', text_norm):
                return opt
            if direction == 'westbound' and re.search(r'\bwest\b', text_norm):
                return opt
            if direction == 'clockwise' and re.search(r'\bclockwise\b', text_norm):
                return opt
            if direction == 'anticlockwise' and (re.search(r'\banticlockwise\b', text_norm) or re.search(r'\bcounter[\s-]?clockwise\b', text_norm)):
                return opt

        # 3) Match by option number anywhere: "1", "first", "one", "option 1", "number 2", "2nd" (word-boundary)
        # Guard: skip numeric matching if the message contains time patterns (e.g. "3pm", "5:30")
        # to prevent "bus times at 3pm" from selecting option 3.
        _has_time_pattern = bool(re.search(r'\d+\s*(am|pm|:\d{2})', text, re.IGNORECASE))
        number_phrases = [
            (1, r'\b(1|one|first|1st)\b'),
            (2, r'\b(2|two|second|2nd)\b'),
            (3, r'\b(3|three|third|3rd)\b'),
            (4, r'\b(4|four|fourth|4th)\b'),
            (5, r'\b(5|five|fifth|5th)\b'),
            (6, r'\b(6|six|sixth|6th)\b'),
            (7, r'\b(7|seven|seventh|7th)\b'),
            (8, r'\b(8|eight|eighth|8th)\b'),
            (9, r'\b(9|nine|ninth|9th)\b'),
            (10, r'\b(10|ten|tenth|10th)\b'),
        ]
        if not _has_time_pattern:
            for idx, pattern in number_phrases:
                if idx <= len(options) and re.search(pattern, text_norm):
                    return options[idx - 1]
        if re.search(r'\boption\s*1\b', text_norm) and len(options) >= 1:
            return options[0]
        if re.search(r'\boption\s*2\b', text_norm) and len(options) >= 2:
            return options[1]
        if re.search(r'\bnumber\s*1\b', text_norm) and len(options) >= 1:
            return options[0]
        if re.search(r'\bnumber\s*2\b', text_norm) and len(options) >= 2:
            return options[1]
        # Standalone digit 1-10 (skip if message contains time patterns)
        if not _has_time_pattern:
            for idx in range(1, min(11, len(options) + 1)):
                if re.search(r'\b' + str(idx) + r'\b', text_norm):
                    return options[idx - 1]

        # 4) Match by station/stop name (for train disambiguation; bus may also match)
        for opt in options:
            name = (opt.get('label') or opt.get('name') or '').strip().lower()
            if not name:
                continue
            # User said the full name, or a substring (e.g. "baker street" matches "Baker Street Underground Station")
            if name in text or text in name:
                return opt
            # Significant words from name all appear in user message (e.g. "baker street" for "Baker Street")
            name_words = [w for w in name.split() if len(w) >= 3 and w not in ('station', 'underground')]
            if name_words and all(w in text for w in name_words):
                return opt

        return None

    def _classify_disambiguation_reply(
        self, user_message: str, options: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]], float]:
        """
        Classify the user's reply when in disambiguation: intent (choose_option, cancel, unclear)
        and confidence for the chosen option (0.0-1.0).
        """
        if not user_message or not options:
            return ('unclear', None, 0.0)

        text = user_message.strip().lower()
        text_norm = ' ' + re.sub(r'\s+', ' ', text) + ' '

        # Cancel intent
        cancel_phrases = [
            r'\bnever\s+mind\b', r'\bcancel\b', r'\bskip\b', r'\bno\s+thanks?\b',
            r'\bforget\s+it\b', r'\bnot\s+now\b', r'\bnone\b', r'\bother\s+stop\b',
            r'\bdifferent\s+stop\b', r'\bactually\s+no\b', r'\bno\s+thanks\b',
        ]
        for pat in cancel_phrases:
            if re.search(pat, text_norm):
                return ('cancel', None, 1.0)

        # Resolve to an option (reuse existing logic)
        selected = self._resolve_disambiguation_reply(user_message, options)
        if selected is None:
            # Fuzzy fallback: best label/towards match by similarity
            best_opt = None
            best_score = 0.0
            for opt in options:
                label = (opt.get('label') or opt.get('name') or '').lower()
                towards = (opt.get('towards') or '').lower()
                platform = (opt.get('platform') or '').lower()
                for candidate in [label, towards, platform]:
                    if not candidate or len(candidate) < 2:
                        continue
                    sim = SequenceMatcher(None, text, candidate).ratio()
                    if sim > best_score:
                        best_score = sim
                        best_opt = opt
            if best_opt is not None and best_score >= 0.5:
                return ('choose_option', best_opt, best_score)
            return ('unclear', None, 0.0)

        # Assign confidence by match type (platform > towards > number)
        confidence = 0.7
        platform = (selected.get('platform') or '').strip().upper()
        towards = (selected.get('towards') or '').strip().lower()
        if platform and re.search(r'\b' + re.escape(platform.lower()) + r'\b', text_norm):
            confidence = 0.95
        elif towards and (towards in text or any(w in text for w in towards.split() if len(w) >= 4)):
            confidence = 0.9
        else:
            # numeric match
            confidence = 0.85
        return ('choose_option', selected, confidence)

    def _reorder_options_by_preference(
        self, options: List[Dict[str, Any]], user_key: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Reorder disambiguation options so last chosen and frequent stops appear first."""
        if not options or not user_key:
            return list(options)
        prefs = self._user_preferences.get(user_key)
        if not prefs:
            return list(options)
        last_id = prefs.get('last_chosen_stop_id')
        frequent = prefs.get('frequent_stops') or {}

        def sort_key(opt: Dict[str, Any]) -> Tuple[int, int]:
            oid = opt.get('id') or ''
            if oid == last_id:
                first = 0
            else:
                first = 1
            count = frequent.get(oid, 0)
            return (first, -count)

        return sorted(options, key=sort_key)

    def _build_timetable_spatial_anchor(
        self, query: str, entities: Dict[str, Any]
    ) -> Optional[SpatialAnchor]:
        """
        Build a SpatialAnchor for timetable disambiguation from LLM-extracted entities.

        Only builds an anchor when the user has explicitly provided geographic context
        via the near_area entity (e.g. "bus times at Lavender Avenue in Kingsbury").

        near_area and towards are intentionally kept separate (for bus AND train):
          - near_area  = where the stop IS (area/neighbourhood context for disambiguation)
          - towards    = where the bus is GOING (direction/destination, used for label
                         matching in _resolve_disambiguation_reply, not for geo-filtering)

        Grounding the query itself as a fallback is deliberately excluded: when there
        is no near_area the query is the ambiguous thing being resolved, so pinning it
        to one Google Places result would silently bias disambiguation against stops in
        other parts of London that share the same street name.
        """
        grounder = get_grounder()
        if not grounder.available:
            return None

        # Only anchor on near_area — an explicit area context provided by the user
        near_area = entities.get('near_area')
        if near_area:
            places_result = grounder.ground(near_area)
            anchor = anchor_from_places(places_result, source="near_area")
            if anchor:
                print(f"[Chatbot] Spatial anchor from near_area='{near_area}': ({anchor.lat}, {anchor.lng})")
                return anchor

        # No near_area → no anchor; let the scoring stage rank without geo-filtering
        return None

    # ------------------------------------------------------------------
    #  Coordinate-based platform direction resolution
    # ------------------------------------------------------------------

    # Compass angles for each TfL direction label
    _DIRECTION_ANGLES: Dict[str, float] = {
        "Northbound": 0.0,
        "Eastbound": 90.0,
        "Southbound": 180.0,
        "Westbound": 270.0,
    }

    def _direction_from_towards_coordinates(
        self,
        towards_target: str,
        timetable_data: Dict[str, Any],
        meaningful_directions: list,
    ) -> Optional[str]:
        """
        Geocode *towards_target* and pick the closest available platform
        direction based on compass bearing from the current station.

        This is the fallback used when no train at the station has the
        target in its destination name (e.g. user says "towards Euston" at
        Wembley Park, but no Metropolitan/Jubilee train lists Euston as a
        destination — Euston is ESE so we pick the closest available
        direction: Southbound).
        """
        import math

        grounder = get_grounder()
        if not grounder or not grounder.available:
            return None

        station_lat = timetable_data.get('stop_lat')
        station_lon = timetable_data.get('stop_lon')
        if station_lat is None or station_lon is None:
            return None

        places_result = grounder.ground(towards_target)
        if not places_result:
            return None
        target_lat = places_result.get('lat')
        target_lon = places_result.get('lng')
        if target_lat is None or target_lon is None:
            return None

        # Compute bearing from station to target
        lat1 = math.radians(float(station_lat))
        lat2 = math.radians(float(target_lat))
        d_lon = math.radians(float(target_lon) - float(station_lon))
        x = math.sin(d_lon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2)
             - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
        bearing = (math.degrees(math.atan2(x, y)) + 360) % 360

        print(
            f"[Chatbot] towards coord fallback: '{towards_target}' bearing={bearing:.0f}° "
            f"from station ({station_lat}, {station_lon})"
        )

        # Find the closest available cardinal direction.
        # Clockwise/Anticlockwise (Circle line) can't be mapped from
        # compass bearing, so skip those.
        best_direction = None
        best_delta = 999.0
        for direction in meaningful_directions:
            angle = self._DIRECTION_ANGLES.get(direction)
            if angle is None:
                continue  # skip Clockwise/Anticlockwise
            # Angular distance (0-180)
            delta = abs(bearing - angle)
            if delta > 180:
                delta = 360 - delta
            if delta < best_delta:
                best_delta = delta
                best_direction = direction

        if best_direction and best_delta <= 135:
            # 135° threshold: pick the closest direction as long as the
            # target isn't almost perpendicular to every available axis.
            print(
                f"[Chatbot] → auto-selected direction '{best_direction}' "
                f"(coordinate fallback, delta={best_delta:.0f}°)"
            )
            return best_direction

        return None

    def _bus_stop_towards_by_bearing(
        self,
        towards_target: str,
        candidates,
    ):
        """
        Determine which candidate bus stop has buses travelling towards
        *towards_target* by comparing each stop's travel bearing (from
        the route sequence) against the bearing to the geocoded target.

        Efficient: only needs ONE stop's travel bearing (2 API calls:
        stop-info + route-sequence).  Since paired bus stops face opposite
        directions, once we know one stop's bearing we know both.

        Returns the winning DisambiguationCandidate, or None on failure.
        """
        import math

        grounder = get_grounder()
        if not grounder or not grounder.available:
            return None

        places_result = grounder.ground(towards_target)
        if not places_result:
            return None
        target_lat = places_result.get('lat')
        target_lon = places_result.get('lng')
        if target_lat is None or target_lon is None:
            return None

        # Only consider candidates that have coordinates
        with_coords = [
            c for c in candidates
            if c.lat is not None and c.lon is not None
        ]
        if len(with_coords) < 2:
            return None

        # Get travel bearing for the FIRST candidate only (efficient).
        # Once we know which way stop A's buses go, we know both directions.
        first = with_coords[0]
        travel_bearing = self.transport_api.get_stop_travel_bearing(first.id)
        if travel_bearing is None:
            # Try the second candidate as fallback
            first = with_coords[1]
            travel_bearing = self.transport_api.get_stop_travel_bearing(first.id)
            if travel_bearing is None:
                return None

        # Identify the "other" stop (typically the opposite side of the road)
        other = next((c for c in with_coords if c.id != first.id), None)
        if other is None:
            return None

        # Helper to compute bearing and great-circle distance from a stop to target
        def _bearing_and_distance(stop_lat, stop_lon):
            lat1 = math.radians(float(stop_lat))
            lon1 = math.radians(float(stop_lon))
            lat2 = math.radians(float(target_lat))
            lon2 = math.radians(float(target_lon))

            d_lon = lon2 - lon1
            x = math.sin(d_lon) * math.cos(lat2)
            y = (math.cos(lat1) * math.sin(lat2)
                 - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
            bearing = (math.degrees(math.atan2(x, y)) + 360) % 360

            # Haversine distance (Earth radius in km, relative comparison only)
            d_lat = lat2 - lat1
            a = (math.sin(d_lat / 2) ** 2
                 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance_km = 6371.0 * c
            return bearing, distance_km

        # Bearing and distance from each candidate to the target
        first_bearing, first_dist = _bearing_and_distance(first.lat, first.lon)
        other_bearing, other_dist = _bearing_and_distance(other.lat, other.lon)

        # Angular difference between travel direction and target direction
        def _angular_delta(travel, target):
            d = abs(travel - target)
            return 360 - d if d > 180 else d

        delta_first = _angular_delta(travel_bearing, first_bearing)
        # Buses at the opposite stop travel ~180° from the first stop
        opposite_travel_bearing = (travel_bearing + 180) % 360
        delta_other = _angular_delta(opposite_travel_bearing, other_bearing)

        print(
            "[Chatbot] bus bearing+distance fallback: "
            f"first='{first.name}' ({first.id}) travel={travel_bearing:.0f}° "
            f"target_bearing={first_bearing:.0f}° delta={delta_first:.0f}° "
            f"dist={first_dist:.2f}km; "
            f"other='{other.name}' ({other.id}) travel_opposite={opposite_travel_bearing:.0f}° "
            f"target_bearing={other_bearing:.0f}° delta={delta_other:.0f}° "
            f"dist={other_dist:.2f}km"
        )

        # Prefer candidates whose travel direction roughly aligns with the target.
        aligned_candidates = []
        if delta_first <= 90:
            aligned_candidates.append(("first", first, first_dist, delta_first))
        if delta_other <= 90:
            aligned_candidates.append(("other", other, other_dist, delta_other))

        chosen = None
        if aligned_candidates:
            # Among aligned candidates, pick the closest to the target.
            chosen_label, chosen, _, _ = min(
                aligned_candidates,
                key=lambda item: item[2],  # distance
            )
            print(
                f"[Chatbot] → selected '{chosen.name}' ({chosen.id}) "
                f"(aligned by bearing, closest by distance)"
            )
            return chosen

        # If neither stop aligns well by bearing, fall back purely to closest distance.
        if first_dist <= other_dist:
            chosen = first
        else:
            chosen = other

        print(
            f"[Chatbot] → selected '{chosen.name}' ({chosen.id}) "
            f"(no good bearing match; closest by distance)"
        )
        return chosen

    def _build_user_context(self, user_key: Optional[str]) -> Optional[UserContext]:
        """Build a UserContext from stored per-user preferences."""
        if not user_key:
            return None
        prefs = self._user_preferences.get(user_key)
        if not prefs:
            return None
        return UserContext(
            last_chosen_stop_id=prefs.get('last_chosen_stop_id'),
            frequent_stops=prefs.get('frequent_stops', {}),
        )

    def _update_user_preference(
        self, user_key: Optional[str], stop_id: str, stop_name: str
    ) -> None:
        """Record that this user chose this stop (for last chosen + frequent stops)."""
        if not user_key or not stop_id:
            return
        prefs = self._user_preferences.setdefault(user_key, {
            'last_chosen_stop_id': None,
            'frequent_stops': {},
        })
        prefs['last_chosen_stop_id'] = stop_id
        prefs['frequent_stops'][stop_id] = prefs['frequent_stops'].get(stop_id, 0) + 1

    def _build_disambiguation_prompt(self, disamb: Dict[str, Any]) -> Dict[str, Any]:
        """Build the disambiguation prompt from stored state (bus direction, train station, or train platform/direction)."""
        options = disamb.get('options', [])
        query = disamb.get('query', '')
        mode = disamb.get('timetable_mode', 'bus')
        is_train_direction = disamb.get('train_direction_disambiguation') is True
        lines = [f"  {i}. {opt.get('label', opt.get('name', 'Unknown'))}" for i, opt in enumerate(options, 1)]
        station_list = '\n'.join(lines[:10])
        if is_train_direction:
            primary = f"Which platform or direction for '{query}'?"
            next_steps = "Reply with the number (e.g. 1 or 2), direction (e.g. Northbound, Southbound), or platform (e.g. Platform 1)."
        elif mode == 'train':
            primary = f"Which train station for '{query}'?"
            next_steps = "Reply with the number (e.g. 1 or 2) or the station name (e.g. Baker Street)."
        else:
            primary = f"Which direction for '{query}'?"
            next_steps = "Reply with anything that includes your choice (e.g. 1 or 2, Stop AA, or the direction name like Willesden)."
        details = f"Choose one:\n{station_list}"
        return {'primary': primary, 'details': details, 'next_steps': next_steps}

    def _format_timetable_response(self, timetable_data: Dict[str, Any], timetable_mode: str) -> Dict[str, Any]:
        """Format timetable data into the same response shape as _handle_timetable success path."""
        stop_name = timetable_data.get('stop_name', '')
        bus_arrivals = timetable_data.get('bus_arrivals', [])
        train_arrivals = timetable_data.get('train_arrivals', [])
        bus_arrivals_by_destination = timetable_data.get('bus_arrivals_by_destination', {})
        train_arrivals_by_direction = timetable_data.get('train_arrivals_by_direction', {})
        bus_grouped = timetable_data.get('bus_arrivals_grouped', {})
        primary_parts = []
        details_parts = []
        if timetable_mode == 'bus' or (timetable_mode == 'both' and bus_arrivals):
            if bus_arrivals:
                primary_parts.append(f"{stop_name} - Next Buses:")
                if bus_grouped:
                    for gid, g in bus_grouped.items():
                        gname = g.get("group_name", stop_name)
                        details_parts.append(f"\n{gname}:")
                        for stop_label in sorted(g.get("stops", {}).keys()):
                            details_parts.append(f"{stop_label}:")
                            for b in g["stops"][stop_label][:6]:
                                details_parts.append(f"  • {b.get('line', 'Unknown')} to {b.get('destination', 'Unknown')}: {b.get('time_minutes', 0)} min")
                else:
                    for bus in bus_arrivals[:10]:
                        details_parts.append(f"  • {bus.get('line', 'Unknown')} to {bus.get('destination', 'Unknown')}: {bus.get('time_minutes', 0)} min")
            else:
                details_parts.append("No buses scheduled in the near future.")
        if timetable_mode == 'train' or (timetable_mode == 'both' and train_arrivals):
            if train_arrivals:
                if primary_parts:
                    details_parts.append("\n")
                primary_parts.append(f"{stop_name} - Next Trains/Tubes:")
                direction_order = ['Northbound', 'Southbound', 'Eastbound', 'Westbound', 'Clockwise', 'Anticlockwise', 'Unknown']
                train_directions = sorted(train_arrivals_by_direction.keys(),
                                         key=lambda x: (direction_order.index(x) if x in direction_order else 999, x))
                for direction in train_directions:
                    direction_trains = train_arrivals_by_direction.get(direction, [])
                    if direction_trains:
                        details_parts.append(f"\n{direction}:")
                        for train in direction_trains[:5]:
                            line = train.get('line', 'Unknown')
                            dest = train.get('destination', 'Unknown')
                            time = train.get('time_minutes', 0)
                            pn = train.get('platform_number', '') or ''
                            if not pn and train.get('platform'):
                                m = re.search(r'platform\s*(\d+)', (train.get('platform') or ''), re.IGNORECASE)
                                pn = m.group(1) if m else ''
                            platform_str = f" (Platform {pn})" if pn else ""
                            details_parts.append(f"  • {line} to {dest}: {time} min{platform_str}")
        primary = "\n".join(primary_parts) if primary_parts else f"Timetable for {stop_name}"
        details = "\n".join(details_parts) if details_parts else None
        return {'primary': primary, 'details': details, 'alternatives': [], 'next_steps': None}
    
    def _generate_response(
        self,
        intent: str,
        entities: Dict[str, str],
        original_message: str,
        user_key: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response based on detected intent and entities."""
        # A) Current conditions
        if intent == 'ask_traffic_status' or intent == 'check_current_conditions':
            return self._handle_traffic_status(entities)
        # G) Public transport disruption
        elif intent == 'ask_transit_disruption':
            resp = self._handle_transit_multimodal(entities, original_message)
            if isinstance(resp, dict) and 'awaiting_line' in resp:
                self.conversation_state['awaiting_disruption_line'] = resp.pop('awaiting_line')
            return resp
        # G2) Transit timetable/times
        elif intent == 'ask_timetable':
            return self._handle_timetable(entities, user_key=user_key, original_message=original_message)
        
        # Existing intents (personalised when username is present)
        elif intent == 'greeting':
            primary = f"Hello{', ' + username if username else ''}! I'm your Smart Traffic Assistant."
            return {
                'primary': primary,
                'details': "I can help you with public transport timetables, service disruptions, and journey planning. What would you like to know?",
                'alternatives': [],
                'next_steps': None
            }
        elif intent == 'goodbye':
            primary = f"Goodbye{', ' + username if username else ''}! Safe travels!"
            return {
                'primary': primary,
                'details': None,
                'alternatives': [],
                'next_steps': None
            }
        
        else:
            return self._handle_unknown_query(entities, original_message)
    
    def _format_response(self, response_data: Dict[str, Any], intent: str, entities: Dict[str, str]) -> str:
        """
        Format response using template: Best option now (route + ETA), Why, Alternatives, Next question
        """
        if isinstance(response_data, str):
            # Legacy string response - wrap it
            return response_data
        
        primary = response_data.get('primary', '')
        details = response_data.get('details')
        alternatives = response_data.get('alternatives', [])
        next_steps = response_data.get('next_steps')
        
        # Build formatted response
        formatted = primary
        
        if details:
            formatted += f"\n\n{details}"
        
        if alternatives:
            formatted += "\n\nAlternatives:"
            for i, alt in enumerate(alternatives, 1):
                formatted += f"\n{i}. {alt}"
        
        if next_steps:
            formatted += f"\n\nNext: {next_steps}"
        
        return formatted
    
    def _handle_traffic_status(self, entities: Dict[str, str]) -> Dict[str, Any]:
        """Handle traffic status queries (A) Current conditions"""
        location = entities.get('location') or entities.get('area')
        route = entities.get('route')
        
        # Check if route is actually a valid transport line name, not a location name
        # Valid tube/transit lines
        valid_routes = ['bakerloo', 'central', 'circle', 'district', 'hammersmith', 'jubilee', 
                       'metropolitan', 'northern', 'piccadilly', 'victoria', 'waterloo', 
                       'dlr', 'overground', 'tram', 'london-overground', 'hammersmith-city', 'waterloo-city']
        
        # If route doesn't look like a valid transit line, treat it as a location instead
        route_lower = route.lower() if route else ''
        is_valid_route = route and any(valid_route in route_lower for valid_route in valid_routes)
        
        # Also check if route contains common query words - if so, it's probably not a real route
        if route and not is_valid_route:
            query_words_in_route = any(word in route_lower for word in ['is', 'the', 'like', 'traffic', 'what', 'how'])
            if query_words_in_route:
                # This is likely a mis-extracted location, not a route
                if not location:
                    location = route
                route = None
        
        if route and is_valid_route:
            # Get traffic data for specific route
            traffic_data = self.transport_api.get_route_traffic(route)
            if traffic_data:
                # Only use data that actually exists - no defaults
                status = traffic_data.get('status')
                delay = traffic_data.get('delay_minutes')
                congestion = traffic_data.get('congestion_level')
                speed = traffic_data.get('average_speed_kmh')
                
                # If critical data is missing, treat as no data
                if status is None and delay is None:
                    return {
                        'primary': f"Traffic data for {route}: Unable to fetch complete data.",
                        'details': "API returned incomplete data. Please try again later.",
                        'alternatives': [],
                        'next_steps': None
                    }
                
                # Use defaults only for display if data exists but some fields are missing
                status = status or 'Status unavailable'
                delay = delay if delay is not None else 0
                
                primary = f"{route}: {status} (delay: {delay} min)" if delay > 0 else f"{route}: {status}"
                details_parts = []
                if congestion:
                    details_parts.append(f"Congestion: {congestion}")
                if speed and speed > 0:
                    details_parts.append(f"Speed: {speed} km/h")
                details = ", ".join(details_parts) if details_parts else None
                
                # Suggest alternatives if there's significant delay
                alternatives = []
                if delay > 15:
                    origin = self.conversation_state.get('origin')
                    destination = self.conversation_state.get('destination')
                    if origin and destination:
                        alternatives.append("Consider alternate route (ask for alternatives)")
                
                return {
                    'primary': primary,
                    'details': details,
                    'alternatives': alternatives,
                    'next_steps': "Need an alternate route?" if delay > 15 else None
                }
            else:
                return {
                    'primary': f"Traffic data for {route}: Unable to fetch at the moment.",
                    'details': "Please try again later or check your mapping app.",
                    'alternatives': [],
                    'next_steps': None
                }
        else:
            # No specific route: ask user to specify one for traffic data
            location = location or 'your area'
            return {
                'primary': f"Traffic in {location}: Data unavailable.",
                'details': "Please specify a route for more detailed information.",
                'alternatives': [],
                'next_steps': "Which route or area are you asking about?"
            }
    
    def _handle_timetable(
        self,
        entities: Dict[str, str],
        user_key: Optional[str] = None,
        original_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle transit timetable queries (bus/train times for TFL stops)."""
        location = entities.get('location') or entities.get('area')
        route = entities.get('route')
        timetable_mode = entities.get('timetable_mode', 'both')  # bus, train, or both
        
        # If we have a canonical CSV stop match, constrain the location to words
        # that actually appear in that stop name. This trims noisy phrases like
        # "get the bus to Lavender Avenue" down to just "Lavender Avenue" before
        # we hit the TfL search API.
        csv_stop_name = entities.get('csv_stop_name')
        if csv_stop_name:
            if not location:
                # Fall back to canonical CSV name when no location was extracted
                location = csv_stop_name
            else:
                # Keep only CSV stop-name words that are present in the extracted location,
                # but only when the user didn't add distinguishing words (e.g. "Frognal Rail"
                # in "Finchley Road and Frognal Rail Station" vs CSV "Finchley Road").
                def _tokenize(text: str) -> List[str]:
                    return re.findall(r"[A-Za-z0-9']+", text)

                csv_tokens = _tokenize(csv_stop_name)
                csv_tokens_lower = {t.lower() for t in csv_tokens}
                loc_tokens = _tokenize(location)
                loc_tokens_lower = {t.lower() for t in loc_tokens}
                # Words in user's location that aren't in the CSV match (e.g. "frognal", "rail")
                loc_extra = [t for t in loc_tokens if t.lower() not in csv_tokens_lower]
                stopwords = {'the', 'to', 'a', 'an'}
                loc_extra_significant = [w for w in loc_extra if len(w) >= 3 and w.lower() not in stopwords]
                # Only truncate when user didn't specify a more specific station name
                if not loc_extra_significant:
                    filtered_tokens = [t for t in csv_tokens if t.lower() in loc_tokens_lower]
                    if filtered_tokens:
                        location = " ".join(filtered_tokens)

        # Persist the cleaned location back into entities so callers and
        # downstream state can see the canonical stop wording.
        if location:
            entities['location'] = location

        # Normalize location using existing method (removes common words, validates)
        if location:
            cleaned_loc, main_words, _ = self.transport_api._normalize_query(location)
            # Validate: ensure we have a meaningful location, but keep the
            # original phrase (e.g. "Finchley Road Underground Station")
            # for the actual TFL search and user-facing messages.
            if not main_words or len(' '.join(main_words)) < 2:
                location = None
        
        if not location:
            return {
                'primary': "I need a location to show the timetable.",
                'details': "Which TFL stop would you like to see bus or train times for?",
                'alternatives': [],
                'next_steps': "For example: 'bus times for Oxford Circus' or 'train times for Wembley Central Station'"
            }
        
        # Build the stop query for the TFL API.
        # For buses, preserve the route number pattern (e.g. "Lavender Avenue 83")
        # so that get_tfl_timetable can distinguish between routes at the same stop.
        stop_query = location
        if route and timetable_mode == 'bus':
            # Avoid duplicating plain street/location names like "High Road high road".
            # Only append the route if it looks meaningfully different from the location,
            # or if it contains a bus route number.
            route_clean = route.strip()
            loc_clean = (location or "").strip()
            has_digits = any(ch.isdigit() for ch in route_clean)
            is_same_phrase = route_clean.lower() == loc_clean.lower()
            if has_digits or not is_same_phrase:
                stop_query = f"{location} {route}"
        
        # Get detailed timetable data for the requested stop
        timetable_data = self.transport_api.get_tfl_timetable(
            stop_query,
            mode_filter=timetable_mode if timetable_mode != 'both' else None
        )
        
        if not timetable_data:
            return {
                'primary': f"Timetable for {location}: Unable to fetch data.",
                'details': "This may not be a TFL stop point, or no arrivals are currently available.",
                'alternatives': [],
                'next_steps': "Try asking about a known London station or stop point."
            }
        
        # Check for error responses
        if isinstance(timetable_data, dict) and 'error' in timetable_data:
            error_type = timetable_data.get('error')
            stop_name = timetable_data.get('stop_name', location)
            query = timetable_data.get('query', location)
            
            if error_type == 'disambiguation_needed':
                stations = timetable_data.get('stations', [])
                count = timetable_data.get('count', len(stations))
                mode = timetable_data.get('mode', '')
                disambiguation_options = timetable_data.get('disambiguation_options') or []

                # ---- Unified Disambiguation Engine ----
                if disambiguation_options:
                    # Convert to DisambiguationCandidate objects
                    candidates = candidates_from_tfl_matches(disambiguation_options)

                    # ---- Towards-location resolution (bus + train) ----
                    # If the user said "towards X", check which candidate stops/stations
                    # actually have vehicles travelling towards X via live arrivals /
                    # route sequence.  This runs before the geo/score engine so it can
                    # hard-filter or auto-resolve without needing a spatial anchor.
                    towards = entities.get('towards') if mode in ('bus', 'train') else None
                    if towards:
                        stop_ids = [c.id for c in candidates if c.id]
                        if mode == 'bus':
                            towards_results = self.transport_api.get_stops_towards_location(
                                stop_ids, towards
                            )
                        else:
                            towards_results = self.transport_api.get_train_stops_towards_location(
                                stop_ids, towards
                            )
                        matching_ids = {
                            sid for sid, r in towards_results.items() if r.get('matches')
                        }
                        print(f"[Chatbot] towards='{towards}' (mode={mode}) matched stop IDs: {matching_ids}")

                        if len(matching_ids) == 1:
                            # Exactly one stop goes towards the target — auto-resolve now,
                            # no need to run the disambiguation engine at all.
                            chosen_id = next(iter(matching_ids))
                            chosen_candidate = next(
                                (c for c in candidates if c.id == chosen_id), None
                            )
                            if chosen_candidate:
                                self._update_user_preference(
                                    user_key, chosen_candidate.id, chosen_candidate.name
                                )
                                print(
                                    f"[Chatbot] towards-resolved '{query}' → "
                                    f"'{chosen_candidate.name}' (towards='{towards}')"
                                )
                                timetable_data = self.transport_api.get_tfl_timetable_by_stop_id(
                                    chosen_candidate.id,
                                    mode_filter=timetable_mode,
                                    stop_name=chosen_candidate.name,
                                )
                                if (
                                    timetable_data
                                    and isinstance(timetable_data, dict)
                                    and 'error' not in timetable_data
                                ):
                                    response_message = self._format_timetable_response(
                                        timetable_data, timetable_mode
                                    )
                                    response_message['timetable_data'] = timetable_data
                                    return response_message
                                # API fetch failed — fall through to present options

                        elif len(matching_ids) > 1:
                            # Multiple stops match — keep only those, let the engine rank them
                            candidates = [c for c in candidates if c.id in matching_ids]

                        # len == 0: no match from arrivals/route-sequence
                        # (quiet hours or target isn't a direct destination).
                        # Bearing fallback: get the travel direction of a
                        # candidate from the route sequence, geocode the
                        # towards target, and pick the stop whose buses
                        # travel in the direction of the target.
                        if len(matching_ids) == 0 and len(candidates) >= 2:
                            closest = self._bus_stop_towards_by_bearing(
                                towards, candidates
                            )
                            if closest:
                                self._update_user_preference(
                                    user_key, closest.id, closest.name
                                )
                                print(
                                    f"[Chatbot] towards coord fallback (bus): "
                                    f"'{towards}' → '{closest.name}' ({closest.id})"
                                )
                                timetable_data = self.transport_api.get_tfl_timetable_by_stop_id(
                                    closest.id,
                                    mode_filter=timetable_mode,
                                    stop_name=closest.name,
                                )
                                if (
                                    timetable_data
                                    and isinstance(timetable_data, dict)
                                    and 'error' not in timetable_data
                                ):
                                    response_message = self._format_timetable_response(
                                        timetable_data, timetable_mode
                                    )
                                    response_message['timetable_data'] = timetable_data
                                    return response_message

                    # Build spatial anchor from near_area entity
                    anchor = self._build_timetable_spatial_anchor(query, entities)

                    # Build user context for preference boosting
                    user_ctx = self._build_user_context(user_key)

                    # Run the disambiguation engine
                    engine = get_disambiguation_engine()
                    result = engine.disambiguate(
                        query=query,
                        candidates=candidates,
                        anchor=anchor,
                        user_context=user_ctx,
                        mode=mode or "bus",
                    )

                    if result.resolved and result.chosen:
                        # Auto-resolved: fetch timetable for the chosen stop directly
                        chosen = result.chosen
                        self._update_user_preference(user_key, chosen.id, chosen.name)
                        print(f"[Chatbot] Auto-resolved '{query}' → '{chosen.name}' (score={result.top_score:.3f})")
                        timetable_data = self.transport_api.get_tfl_timetable_by_stop_id(
                            chosen.id, mode_filter=timetable_mode, stop_name=chosen.name
                        )
                        if timetable_data and isinstance(timetable_data, dict) and 'error' not in timetable_data:
                            response_message = self._format_timetable_response(timetable_data, timetable_mode)
                            response_message['timetable_data'] = timetable_data
                            return response_message
                        # If fetch failed, fall through to present options

                    if result.action == "ask_rephrase":
                        mode_text = 'train stations' if mode == 'train' else 'bus stops' if mode == 'bus' else 'stops'
                        return {
                            'primary': f"I couldn't confidently match '{query}' to a specific {mode_text.rstrip('s')}.",
                            'details': "Could you be more specific?",
                            'alternatives': [],
                            'next_steps': f"Try including a direction, area, or route number (e.g., '{query} towards Wembley' or '{query} in Kingsbury')."
                        }

                    # present_options: store ranked candidates for user to choose from
                    ranked_options = [c.to_dict() for c in result.candidates]
                    # Also apply legacy preference reordering for bus stops
                    if mode == 'bus':
                        ranked_options = self._reorder_options_by_preference(ranked_options, user_key)

                    self.conversation_state['timetable_disambiguation'] = {
                        'options': ranked_options,
                        'query': query,
                        'timetable_mode': timetable_mode or mode
                    }
                    # FSM transition: enter disambiguation state
                    self.state_tracker.start_disambiguation(
                        user_key or "_anon",
                        options=ranked_options,
                        query=query,
                        mode=timetable_mode or mode,
                    )
                    prompt = self._build_disambiguation_prompt(self.conversation_state['timetable_disambiguation'])
                    return {
                        'primary': prompt['primary'],
                        'details': prompt.get('details'),
                        'alternatives': [],
                        'next_steps': prompt.get('next_steps')
                    }

                # No disambiguation_options (legacy/fallback): just list options
                if stations:
                    station_list = '\n'.join([f"  • {station}" for station in stations[:10]])
                    if count > 10:
                        station_list += f"\n  ... and {count - 10} more"
                    if mode == 'train':
                        primary_msg = f"Multiple train stations found for '{query}'. Which one did you mean?"
                        details_msg = f"Found {count} train station(s):\n{station_list}"
                        next_steps_msg = f"Please specify the exact stop name (e.g., '{stations[0]}' or '{stations[1] if len(stations) > 1 else stations[0]}')."
                    elif mode == 'bus':
                        primary_msg = f"Multiple bus stops found for '{query}'. Which one did you mean?"
                        details_msg = f"Found {count} bus stop(s):\n{station_list}"
                        next_steps_msg = (
                            "Please specify the bus route number "
                            f"(e.g., 'bus times for {query} route 83' or 'bus times for {query} route 302'). "
                            "This will show all stops serving that route."
                        )
                    else:
                        primary_msg = f"Multiple stops found for '{query}'. Which one did you mean?"
                        details_msg = f"Found {count} stop(s):\n{station_list}"
                        next_steps_msg = f"Please specify the exact stop name (e.g., '{stations[0]}' or '{stations[1] if len(stations) > 1 else stations[0]}')."
                    return {
                        'primary': primary_msg,
                        'details': details_msg,
                        'alternatives': [],
                        'next_steps': next_steps_msg
                    }
                else:
                    mode_text = 'train stations' if mode == 'train' else 'bus stops' if mode == 'bus' else 'stops'
                    return {
                        'primary': f"Multiple {mode_text} found for '{query}'.",
                        'details': "Please specify which stop you're looking for.",
                        'alternatives': [],
                        'next_steps': "Try being more specific with the stop name."
                    }
            elif error_type == 'route_not_served':
                stop_name = timetable_data.get('stop_name', query)
                requested_route = timetable_data.get('requested_route', '')
                available_routes_str = timetable_data.get('available_routes_str', '')
                available_routes = timetable_data.get('available_routes', [])
                
                # Get first available route for example, or use placeholder
                example_route = available_routes[0] if available_routes else '[route number]'
                
                return {
                    'primary': f"Route {requested_route} doesn't serve {stop_name}.",
                    'details': f"Available routes at this stop: {available_routes_str}",
                    'alternatives': [],
                    'next_steps': f"Try asking for one of the available routes (e.g., 'bus times for {stop_name} {example_route}')."
                }
            elif error_type == 'not_found':
                mode = timetable_data.get('mode', '')
                if mode == 'train':
                    return {
                        'primary': f"'{query}' does not have train services.",
                        'details': "This location may be a bus stop or may not be a TFL stop point with train/tube services.",
                        'alternatives': [],
                        'next_steps': "Try asking about a known London train/tube station (e.g., 'Oxford Circus', 'King's Cross', 'Paddington', 'Neasden')."
                    }
                else:
                    return {
                        'primary': f"'{query}' is not a recognized TFL stop point.",
                        'details': "I couldn't find this location in the TFL database.",
                        'alternatives': [],
                        'next_steps': "Try asking about a known London station or stop point (e.g., 'Oxford Circus', 'King's Cross', 'Paddington')."
                    }
            elif error_type == 'no_arrivals':
                return {
                    'primary': f"No arrivals currently available for {stop_name}.",
                    'details': f"{stop_name} is a valid TFL stop point, but there are no upcoming arrivals at this time.",
                    'alternatives': [],
                    'next_steps': "This could be due to service disruptions, late night hours, or the stop not being in service. Try again later or check another stop."
                }
        
        bus_arrivals = timetable_data.get('bus_arrivals', [])
        train_arrivals = timetable_data.get('train_arrivals', [])
        bus_arrivals_by_destination = timetable_data.get('bus_arrivals_by_destination', {})
        train_arrivals_by_direction = timetable_data.get('train_arrivals_by_direction', {})
        stop_name = timetable_data.get('stop_name', location)
        
        # Train platform/direction disambiguation: if multiple directions (or platforms), ask user to choose
        # Unless the query already specifies direction or platform – then show that one immediately
        if (timetable_mode == 'train' or (timetable_mode == 'both' and train_arrivals)) and train_arrivals_by_direction:
            direction_order = ['Northbound', 'Southbound', 'Eastbound', 'Westbound', 'Clockwise', 'Anticlockwise', 'Unknown']
            meaningful_directions = [
                d for d in train_arrivals_by_direction.keys()
                if d and d != 'Unknown'
            ]
            if not meaningful_directions:
                meaningful_directions = [d for d in train_arrivals_by_direction.keys() if d]
            if len(meaningful_directions) >= 2:
                chosen_direction = None
                if original_message:
                    text = original_message.strip().lower()
                    # Build direction -> platform number for this stop (same as options below)
                    direction_platforms = {}
                    for direction in meaningful_directions:
                        direction_trains = train_arrivals_by_direction.get(direction, [])
                        first_train = direction_trains[0] if direction_trains else {}
                        platform_num = (first_train.get('platform_number') or '').strip()
                        if not platform_num and first_train.get('platform'):
                            m = re.search(r'platform\s*(\d+)', (first_train.get('platform') or ''), re.IGNORECASE)
                            platform_num = m.group(1) if m else ''
                        direction_platforms[direction] = platform_num
                    # 1) Match "platform N" in query
                    for direction, platform_num in direction_platforms.items():
                        if platform_num and re.search(r'\bplatform\s*' + re.escape(platform_num) + r'\b', text):
                            chosen_direction = direction
                            break
                    # 2) Match direction words: northbound, southbound, westbound, eastbound, etc.
                    if not chosen_direction:
                        for direction in meaningful_directions:
                            d_lower = direction.lower()
                            if d_lower in text:
                                chosen_direction = direction
                                break
                            if direction == 'Northbound' and re.search(r'\bnorth\b', text):
                                chosen_direction = direction
                                break
                            if direction == 'Southbound' and re.search(r'\bsouth\b', text):
                                chosen_direction = direction
                                break
                            if direction == 'Eastbound' and re.search(r'\beast\b', text):
                                chosen_direction = direction
                                break
                            if direction == 'Westbound' and re.search(r'\bwest\b', text):
                                chosen_direction = direction
                                break
                            if direction == 'Clockwise' and re.search(r'\bclockwise\b', text):
                                chosen_direction = direction
                                break
                            if direction == 'Anticlockwise' and (re.search(r'\banticlockwise\b', text) or re.search(r'\bcounter[\s-]?clockwise\b', text)):
                                chosen_direction = direction
                                break
                    # 3) Match by 'towards' entity (or 'destination' when Haiku puts
                    #    "timetable for X to Y" destination in that field).
                    #    3a) Check destination names for target tokens.
                    #    3b) Fallback: geocode the towards target and pick the
                    #        compass direction (N/S/E/W) from station to target.
                    if not chosen_direction:
                        towards_target = (
                            entities.get('towards')
                            or entities.get('destination')
                        )
                        if towards_target:
                            # 3a) Destination-name matching
                            target_tokens = self.transport_api._target_tokens(towards_target)
                            if target_tokens:
                                for direction in meaningful_directions:
                                    direction_trains = train_arrivals_by_direction.get(direction, [])
                                    for train in direction_trains:
                                        dest = train.get('destination', '') or ''
                                        if self.transport_api._text_contains_target(dest, target_tokens):
                                            chosen_direction = direction
                                            print(
                                                f"[Chatbot] towards/dest '{towards_target}' "
                                                f"→ auto-selected direction '{direction}' (name match)"
                                            )
                                            break
                                    if chosen_direction:
                                        break

                            # 3b) Coordinate-based fallback: geocode towards target,
                            #     compute bearing from station, map to direction.
                            if not chosen_direction:
                                chosen_direction = self._direction_from_towards_coordinates(
                                    towards_target, timetable_data, meaningful_directions
                                )
                if chosen_direction:
                    # Query already specified direction/platform – filter and show immediately (no disambiguation)
                    filtered_trains = [t for t in train_arrivals if (t.get('direction') or '') == chosen_direction]
                    train_arrivals = filtered_trains[:10]
                    train_arrivals_by_direction = {chosen_direction: filtered_trains[:10]}
                    timetable_data = {
                        **timetable_data,
                        'train_arrivals': train_arrivals,
                        'train_arrivals_by_direction': train_arrivals_by_direction,
                    }
                else:
                    # No direction/platform in query – show disambiguation
                    train_direction_options = []
                    for direction in sorted(
                        meaningful_directions,
                        key=lambda x: (direction_order.index(x) if x in direction_order else 999, x)
                    ):
                        direction_trains = train_arrivals_by_direction.get(direction, [])
                        first_train = direction_trains[0] if direction_trains else {}
                        platform_num = (first_train.get('platform_number') or '').strip()
                        if not platform_num and first_train.get('platform'):
                            m = re.search(r'platform\s*(\d+)', (first_train.get('platform') or ''), re.IGNORECASE)
                            platform_num = m.group(1) if m else ''
                        label = f"{direction} (Platform {platform_num})" if platform_num else direction
                        train_direction_options.append({
                            'id': direction,
                            'name': direction,
                            'label': label,
                            'direction': direction,
                            'platform': platform_num or None,
                            # Reuse stop-level coordinates when available so the frontend
                            # can still show the station on a map for platform choices.
                            'lat': timetable_data.get('stop_lat'),
                            'lon': timetable_data.get('stop_lon'),
                        })
                    self.conversation_state['timetable_disambiguation'] = {
                        'options': train_direction_options,
                        'query': stop_name,
                        'timetable_mode': 'train',
                        'train_direction_disambiguation': True,
                        'timetable_data': timetable_data,
                    }
                    # FSM transition: enter disambiguation state
                    self.state_tracker.start_disambiguation(
                        user_key or "_anon",
                        options=train_direction_options,
                        query=stop_name,
                        mode='train',
                    )
                    prompt = self._build_disambiguation_prompt(self.conversation_state['timetable_disambiguation'])
                    return {
                        'primary': prompt['primary'],
                        'details': prompt.get('details'),
                        'alternatives': [],
                        'next_steps': prompt.get('next_steps')
                    }
        
        # Format the timetable response
        primary_parts = []
        details_parts = []
        
        if timetable_mode == 'bus' or (timetable_mode == 'both' and bus_arrivals):
            if bus_arrivals:
                primary_parts.append(f"{stop_name} - Next Buses:")
                
                bus_grouped = timetable_data.get("bus_arrivals_grouped", {})
                if bus_grouped:
                    # If your search returns a single group, this prints nicely under the same heading.
                    # If multiple groups match, you’ll see each group name as a sub-heading.
                    for gid, g in bus_grouped.items():
                        gname = g.get("group_name", stop_name)
                        details_parts.append(f"\n{gname}:")
                        stops = g.get("stops", {})

                        for stop_label in sorted(stops.keys()):
                            details_parts.append(f"{stop_label}:")
                            for b in stops[stop_label][:6]:  # limit per stop letter/stand
                                line = b.get("line", "Unknown")
                                dest = b.get("destination", "Unknown")
                                t = b.get("time_minutes", 0)
                                details_parts.append(f"  • {line} to {dest}: {t} min")
                else:
                    # Fallback to old behaviour (grouped by destination)
                    if bus_arrivals_by_destination:
                        for destination in sorted(bus_arrivals_by_destination.keys()):
                            destination_buses = bus_arrivals_by_destination[destination]
                            if destination_buses:
                                details_parts.append(f"\n{destination}:")
                                bus_lines = []
                                for bus in destination_buses:
                                    line = bus.get('line', 'Unknown')
                                    time = bus.get('time_minutes', 0)
                                    bus_lines.append(f"  • {line}: {time} min")
                                details_parts.append("\n".join(bus_lines))
                    else:
                        bus_lines = []
                        for bus in bus_arrivals[:10]:
                            line = bus.get('line', 'Unknown')
                            destination = bus.get('destination', 'Unknown')
                            time = bus.get('time_minutes', 0)
                            bus_lines.append(f"  • {line} to {destination}: {time} min")
                        details_parts.append("\n".join(bus_lines))
            else:
                details_parts.append("No buses scheduled in the near future.")
        
        if timetable_mode == 'train' or (timetable_mode == 'both' and train_arrivals):
            if train_arrivals:
                if timetable_mode == 'both' and bus_arrivals:
                    details_parts.append("\n")  # Separator
                primary_parts.append(f"{stop_name} - Next Trains/Tubes:")
                
                # Sort directions for consistent display (prioritize Northbound/Southbound)
                direction_order = ['Northbound', 'Southbound', 'Eastbound', 'Westbound', 'Clockwise', 'Anticlockwise', 'Unknown']
                train_directions = sorted(train_arrivals_by_direction.keys(), 
                                         key=lambda x: (direction_order.index(x) if x in direction_order else 999, x))
                
                if len(train_directions) > 1 or (len(train_directions) == 1 and train_directions[0] != 'Unknown'):
                    # Grouped by direction
                    for direction in train_directions:
                        direction_trains = train_arrivals_by_direction[direction]
                        if direction_trains:
                            details_parts.append(f"\n{direction}:")
                            train_lines = []
                            for train in direction_trains[:5]:  # Show next 5 per direction
                                line = train.get('line', 'Unknown')
                                destination = train.get('destination', 'Unknown')
                                time = train.get('time_minutes', 0)
                                platform_num = train.get('platform_number', '') or ''
                                if not platform_num and train.get('platform'):
                                    m = re.search(r'platform\s*(\d+)', (train.get('platform') or ''), re.IGNORECASE)
                                    platform_num = m.group(1) if m else ''
                                platform_str = f" (Platform {platform_num})" if platform_num else ""
                                train_lines.append(f"  • {line} to {destination}: {time} min{platform_str}")
                            details_parts.append("\n".join(train_lines))
                else:
                    # Not grouped, show flat list
                    train_lines = []
                    for train in train_arrivals[:10]:  # Show next 10 trains
                        line = train.get('line', 'Unknown')
                        destination = train.get('destination', 'Unknown')
                        time = train.get('time_minutes', 0)
                        platform_num = train.get('platform_number', '') or ''
                        if not platform_num and train.get('platform'):
                            m = re.search(r'platform\s*(\d+)', (train.get('platform') or ''), re.IGNORECASE)
                            platform_num = m.group(1) if m else ''
                        platform_str = f" (Platform {platform_num})" if platform_num else ""
                        train_lines.append(f"  • {line} to {destination}: {time} min{platform_str}")
                    details_parts.append("\n".join(train_lines))
            else:
                details_parts.append("No trains/tubes scheduled in the near future.")
        
        if not bus_arrivals and not train_arrivals:
            return {
                'primary': f"No arrivals available for {stop_name}.",
                'details': "Check back later or verify this is an active TFL stop point.",
                'alternatives': [],
                'next_steps': None
            }
        
        primary = "\n".join(primary_parts) if primary_parts else f"Timetable for {stop_name}"
        details = "\n".join(details_parts) if details_parts else None
        
        return {
            'primary': primary,
            'details': details,
            'alternatives': [],
            'next_steps': None,
            'timetable_data': timetable_data,
        }
    
    def _handle_transit_multimodal(self, entities: Dict[str, str], original_message: str = '') -> Dict[str, Any]:
        """Handle public transport disruption and multimodal queries.
        Uses NLP extraction from original_message for train line (from _train_lines) or bus route (from tfl_bus_routes.txt).
        """
        location = entities.get('location')
        route = entities.get('route')
        line = entities.get('line') or route
        bus_route_pattern = r'\b([Nn]?\d{1,3})\b'
        is_bus_route = False

        # Prefer new extraction from original message when available
        if original_message and hasattr(self.nlp, 'extract_train_disruption_line') and hasattr(self.nlp, 'extract_bus_disruption_route'):
            train_line = self.nlp.extract_train_disruption_line(original_message)
            bus_route = self.nlp.extract_bus_disruption_route(original_message)
            if train_line:
                line = train_line
                is_bus_route = False
            elif bus_route:
                line = bus_route
                is_bus_route = True
            else:
                # No valid line from extraction: don't use entities as line (e.g. "status on fake line" as location)
                line = None
                is_bus_route = False
        else:
            is_bus_route = bool(line and re.search(bus_route_pattern, line))

        # No line identified for a disruption/status query: return error and allow follow-up with just line/route
        if not line:
            msg_lower = (original_message or '').lower()
            if 'bus' in msg_lower and not any(x in msg_lower for x in ['train', 'tube', 'line', 'dlr', 'overground', 'underground']):
                return {
                    'primary': "I couldn't find a bus route.",
                    'details': "Please specify a valid TfL bus route number (e.g. 83, N29). You can reply with just the route number.",
                    'alternatives': [],
                    'next_steps': None,
                    'awaiting_line': 'bus',
                }
            return {
                'primary': "I couldn't find a train line.",
                'details': "Please specify a London Underground, Overground or DLR line by name (e.g. Victoria, Windrush, Northern, DLR), or a bus route (e.g. bus 83). You can reply with just the line or route.",
                'alternatives': [],
                'next_steps': None,
                'awaiting_line': 'train',
            }

        # Check for transit disruption (train/tube) or bus disruption
        if line:
            disruption = None
            if is_bus_route:
                disruption = self.transport_api.get_bus_disruption(line)
            else:
                disruption = self.transport_api.get_transit_disruption(line)

            # Train/tube: handle API error responses
            if not is_bus_route and isinstance(disruption, dict) and disruption.get('error'):
                err = disruption.get('error')
                if err == 'line_not_found':
                    return {
                        'primary': "I couldn't find a train line.",
                        'details': "Please specify a London Underground, Overground or DLR line by name (e.g. Victoria, Windrush, Northern, DLR).",
                        'alternatives': [],
                        'next_steps': None
                    }
                if err == 'api_error':
                    return {
                        'primary': "I couldn't retrieve the status for that line right now.",
                        'details': "Please try again later.",
                        'alternatives': [],
                        'next_steps': None
                    }

            if disruption and not disruption.get('error'):
                # Use the actual line/route name from the disruption data if available
                if is_bus_route:
                    actual_route = disruption.get('route', line)
                    display_route = f"Bus {actual_route}" if not actual_route.startswith('Bus') else actual_route
                else:
                    actual_line = disruption.get('line', line)
                    # Add "Line" if not already present
                    display_route = f"{actual_line} Line" if "Line" not in actual_line else actual_line
                
                status = disruption.get('status', 'Disruption reported')
                description = disruption.get('description', '')
                
                # Get affected locations/segments
                affected_locations = disruption.get('affected_locations', [])
                # Filter out "Entire Line" from affected locations
                affected_locations = [loc for loc in affected_locations if loc != "Entire Line"]
                
                # Build status message with location information
                if affected_locations:
                    # Use first location (most relevant) or join if multiple
                    location_info = affected_locations[0] if len(affected_locations) == 1 else ', '.join(affected_locations[:2])
                    status_with_location = f"{status} {location_info}"
                else:
                    status_with_location = status
                
                # Only show description if it's different from status and not empty
                if description and description != status and description.strip():
                    return {
                        'primary': f"{display_route}: {status_with_location}",
                        'details': description,
                        'alternatives': disruption.get('alternatives', []),
                        'next_steps': None,
                        'disruption': disruption,
                    }
                else:
                    # If description is same as status or empty, just show status
                    return {
                        'primary': f"{display_route}: {status_with_location}",
                        'details': None,
                        'alternatives': disruption.get('alternatives', []),
                        'next_steps': None,
                        'disruption': disruption,
                    }
            else:
                # No disruption found, but we have a line/route
                if is_bus_route:
                    display_route = f"Bus {line}" if not line.startswith('Bus') else line
                    return {
                        'primary': f"Checking bus status for {display_route}...",
                        'details': "No disruptions reported currently.",
                        'alternatives': [],
                        'next_steps': None
                    }
                else:
                    # Fetch actual line name from API
                    route_data = self.transport_api.get_route_traffic(line)
                    actual_line = route_data.get('route', line) if route_data else line
                    # Add "Line" if not already present
                    display_line = f"{actual_line} Line" if "Line" not in actual_line else actual_line
                    
                    return {
                        'primary': f"Checking transit status for {display_line}...",
                        'details': "No disruptions reported currently.",
                        'alternatives': [],
                        'next_steps': None
                    }
        
        # Multimodal route comparison (drive vs transit)
        origin = entities.get('origin') or self.conversation_state.get('origin')
        destination = entities.get('destination') or self.conversation_state.get('destination')
        
        if origin and destination:
            drive_route = self.transport_api.get_route_recommendation(origin, destination)
            transit_route = self.transport_api.get_transit_route(origin, destination)
            
            if drive_route and transit_route:
                drive_time = drive_route.get('duration_minutes', 0)
                transit_time = transit_route.get('duration_minutes', 0)
                
                if transit_time < drive_time:
                    primary = f"Public transport faster: {transit_time} min vs {drive_time} min driving"
                else:
                    primary = f"Driving faster: {drive_time} min vs {transit_time} min transit"
                
                details = f"Drive: {drive_time} min | Transit: {transit_time} min"
                
                # Suggest park-and-ride if applicable
                alternatives = []
                if transit_time < drive_time * 1.5:  # Transit not much slower
                    alternatives.append(f"Consider park-and-ride: drive part way, then take transit")
                
                return {
                    'primary': primary,
                    'details': details,
                    'alternatives': alternatives,
                    'next_steps': None
                }
        
        return {
            'primary': f"Checking transit status for {location or 'your area'}...",
            'details': "No disruptions reported currently.",
            'alternatives': [],
            'next_steps': None
        }
    
    def _handle_unknown_query(self, entities: Dict[str, str], original_message: str) -> Dict[str, Any]:
        """Handle queries that don't match known intents"""
        # Try to extract any location/route info and provide generic response
        location = entities.get('location')
        route = entities.get('route')
        origin = entities.get('origin')
        destination = entities.get('destination')
        
        if location or route or origin or destination:
            # Prefer journey-style slots for the target if present.
            target = destination or origin or location or route
            return {
                'primary': f"I didn't fully understand your question about {target}.",
                'details': "Please rephrase and try asking about public transport timetables, service disruptions, or journey planning.",
                'alternatives': [],
                'next_steps': "What would you like to know about " + target + "?"
            }
        else:
            return {
                'primary': "I didn't understand that request.",
                'details': "Please rephrase your question. I can help with public transport timetables, service disruptions, and journey planning.",
                'alternatives': [],
                'next_steps': "Try asking: 'Bus times for Oxford Circus', 'Is the Northern line running?', or 'Plan a journey from Neasden to Oxford Circus.'"
            }
