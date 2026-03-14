"""
Smart Traffic Query Assistant - Main Chatbot Core
Handles NLP processing, traffic data fetching, and ML predictions
"""
from nlp_processor import NLPProcessor
from transport_api import TransportDataFetcher
import json
import re
from typing import Dict, Optional, Any, List, Tuple

class TrafficChatbot:
    def __init__(self):
        """Initialize chatbot with NLP, API, and ML components"""
        self.nlp = NLPProcessor()
        self.transport_api = TransportDataFetcher()
        # Conversation state - track ongoing context
        self.conversation_state: Dict[str, Any] = {
            'origin': None,
            'destination': None,
            # Dialogue state for bus stop direction disambiguation (DST)
            'timetable_disambiguation': None,  # when set: { 'options': [...], 'query': str, 'timetable_mode': str }
            # When we asked "couldn't find a train/bus" - user can reply with just a line/route name
            'awaiting_disruption_line': None,  # 'train' | 'bus' when waiting for follow-up line/route
        }
        # Per-user preference tracking removed in this prototype
        print("Traffic Chatbot initialized successfully")
    
    def process_message(
        self,
        user_message: str,
        user_key: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process user message and generate response

        Args:
            user_message: User's natural language query
            user_key: Optional per-user/session key for preferences (e.g. username or '_anon').
            username: Optional display name for personalised replies (e.g. greeting, goodbye).

        Returns:
            dict: Response with message, intent, entities, and confidence
        """
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
            # One significant word from towards (e.g. "willesden" from "Willesden Bus Garage")
            for word in towards.split():
                if len(word) >= 4 and word in text:  # avoid "to", "the"
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
        number_phrases = [
            (1, r'\b(1|one|first|1st)\b'),
            (2, r'\b(2|two|second|2nd)\b'),
            (3, r'\b(3|three|third|3rd)\b'),
            (4, r'\b(4|four|fourth|4th)\b'),
            (5, r'\b(5|five|fifth|5th)\b'),
        ]
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
        # Standalone digit 1-10
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
        # G) Public transport disruption + multimodal
        elif intent == 'ask_transit_disruption' or intent == 'ask_multimodal':
            resp = self._handle_transit_multimodal(entities, original_message)
            if isinstance(resp, dict) and 'awaiting_line' in resp:
                self.conversation_state['awaiting_disruption_line'] = resp.pop('awaiting_line')
            return resp
        # G2) Transit timetable/times
        elif intent == 'ask_timetable' or intent == 'ask_transit_times':
            return self._handle_timetable(entities, user_key=user_key, original_message=original_message)
        
        # Existing intents (personalised when username is present)
        elif intent == 'greeting':
            primary = f"Hello{', ' + username if username else ''}! I'm your Smart Traffic Assistant."
            return {
                'primary': primary,
                'details': "I can help you with traffic status, delays, routes, incidents, closures, parking, and more. What would you like to know?",
                'alternatives': [],
                'next_steps': None
            }
        elif intent == 'goodbye':
            primary = f"Goodbye{', ' + username if username else ''}! Drive safely!"
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
        
        # Persist the cleaned location back into entities
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
                
                # Bus or train disambiguation: store state and handle replies
                if disambiguation_options:
                    self.conversation_state['timetable_disambiguation'] = {
                        'options': disambiguation_options,
                        'query': query,
                        'timetable_mode': timetable_mode or mode
                    }
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
        
        # Use shared formatter
        result = self._format_timetable_response(timetable_data, timetable_mode)
        result['timetable_data'] = timetable_data
        return result
    
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
        
        if location or route:
            target = location or route
            return {
                'primary': f"I didn't fully understand your question about {target}.",
                'details': "Please rephrase and try asking about traffic status, delays, routes, incidents, closures, congestion, or public transport.",
                'alternatives': [],
                'next_steps': "What would you like to know about " + target + "?"
            }
        else:
            return {
                'primary': "I didn't understand that request.",
                'details': "Please rephrase your question. I can help with traffic status, delays, routes, incidents, closures, congestion, transit disruptions, timetables, and more.",
                'alternatives': [],
                'next_steps': "Try asking: 'What's the traffic like on Highway 101?' or 'Fastest route from A to B?'"
            }
