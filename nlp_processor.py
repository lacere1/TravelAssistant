"""
NLP Processor using Hugging Face Transformers
Handles intent detection and entity extraction
"""
from transformers import pipeline
import re
import csv
import os
from difflib import SequenceMatcher
from typing import Dict, List, Any, Set, Optional, Tuple


class NLPProcessor:
    def __init__(self):
        """Initialize NLP models for intent classification and entity extraction"""
        print("Loading NLP models...")


        self._synonym_expansion_enabled = False
        


        self.intent_classifier = None
        
        # Intent labels for classification
        self.intent_labels = [
            "check_current_conditions",
            "ask_traffic_status",
            "ask_delay",
            "ask_transit_disruption",
            "ask_multimodal",
            "ask_timetable",
            "ask_transit_times",
            "ask_congestion",
            "greeting",
            "goodbye"
        ]
        
        print("NLP models loaded successfully")
        
        # Pre-load stop name datasets for CSV-backed intent refinement
        self._bus_stops: List[str] = []
        self._train_stations: List[str] = []
        self._train_lines: List[str] = []  # London Underground, Overground, DLR line names
        self._bus_routes: Set[str] = set()  # TfL bus route ids from tfl_bus_routes.txt
        self._stops_loaded: bool = False
        self._load_stop_datasets()
    
    def _expand_with_synonyms(self, tokens: List[str]) -> Set[str]:
        """
        Expand a list of tokens with WordNet synonyms.
        
        Returns a set containing the original tokens plus any synonym words
        (multi-word synonyms are split into individual tokens).
        If NLTK/WordNet are unavailable, this simply returns the original tokens.
        """


        return set(t.lower() for t in tokens if t)
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process user input to extract intent and entities
        
        Args:
            text: User's natural language input
            
        Returns:
            dict: Contains intent, entities, and confidence score
        """
        original_text = text
        # Normalize text
        text = text.strip().lower()
        
        # Intent classification
        intent, confidence = self._classify_intent(text)
        
        # Entity extraction (primary NER + slot filling)
        entities = self._extract_entities(text)
        
        # Secondary, CSV-backed NER + intent refinement based on stop names
        intent, entities, confidence = self._refine_with_stop_datasets(
            original_text, intent, entities, confidence
        )
        
        return {
            'intent': intent,
            'entities': entities,
            'confidence': confidence
        }
    
    def _classify_intent(self, text: str) -> tuple:
        """Classify user intent using transformer model or fallback to rule-based"""
        
        # First check rule-based patterns for common traffic queries to avoid transformer misclassification
        # Check for timetable queries FIRST (before event travel)
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ['bus times', 'train times', 'tube times', 'bus or train times', 'train or bus times', 'timetable', 'next bus', 'next train', 'when is the next']):
            return 'ask_timetable', 0.9
        
        # Explicitly handle disruption questions (train line or bus route) before using the transformer
        if any(kw in text_lower for kw in ['disruption', 'disrupted', 'status', 'delay', 'delays']):
            has_train = any(
                x in text_lower for x in [
                    ' line', ' tube', 'train', 'overground', 'dlr', 'underground',
                    'bakerloo', 'central', 'circle', 'district', 'hammersmith', 'jubilee',
                    'metropolitan', 'northern', 'piccadilly', 'victoria', 'waterloo',
                    'windrush', 'lioness', 'mildmay', 'suffragette', 'weaver', 'liberty'
                ]
            )
            has_bus_route = bool(re.search(r'bus.*\d|\d.*bus', text_lower))
            has_bus_status = 'bus' in text_lower  # e.g. "bus status", "get bus status for it"
            if has_train or has_bus_route or has_bus_status:
                return 'ask_transit_disruption', 0.9
        
        # Check for "what is the traffic like" pattern explicitly before using transformer
        if any(phrase in text_lower for phrase in ['what is the traffic like', 'what\'s the traffic like', 'how is the traffic', 'traffic like']):
            return 'ask_traffic_status', 0.9
        
        # Check for "traffic in [location]" pattern (e.g., "traffic in wembley high road")
        # This matches "traffic in" followed by one or more words
        if re.search(r'traffic\s+in\s+[a-z]+(?:\s+[a-z]+)*', text_lower):
            return 'ask_traffic_status', 0.9
        
        if self.intent_classifier:
            try:
                result = self.intent_classifier(text, self.intent_labels)
                intent = result['labels'][0]
                confidence = result['scores'][0]

                # If overall confidence is low, fall back to synonym-aware rule-based intent
                # instead of trusting the transformer classification.
                if confidence < 0.6:
                    rule_intent, rule_confidence = self._rule_based_intent(text)
                    return rule_intent, rule_confidence
                
                # If transformer gives low confidence and it's a questionable classification,
                # prefer rule-based for traffic status queries
                if confidence < 0.5 and 'traffic' in text_lower:
                    rule_intent, rule_confidence = self._rule_based_intent(text)
                    if rule_confidence > 0.8:
                        return rule_intent, rule_confidence
                
                return intent, confidence
            except Exception as e:
                print(f"Intent classification error: {e}, using fallback")
        
        # Fallback to rule-based intent detection
        return self._rule_based_intent(text)
    
    def _rule_based_intent(self, text: str) -> tuple:
        """Rule-based intent detection as fallback - covers all intents A-K"""
        text_lower = text.lower()
        # Tokenize and expand with WordNet synonyms (if available)
        tokens = re.findall(r'\w+', text_lower)
        expanded_words = self._expand_with_synonyms(tokens)

        # Concept sets for synonym-aware matching
        traffic_concepts = {'traffic', 'congestion', 'jam'}
        greeting_concepts = {'hello', 'hi', 'hey', 'greetings'}
        goodbye_concepts = {'bye', 'goodbye', 'farewell'}
        disruption_concepts = {'disruption', 'disturbance', 'interruption'}
        transit_concepts = {'transit', 'train', 'tube', 'subway', 'metro', 'tram', 'bus'}
        timetable_concepts = {'timetable', 'schedule', 'times', 'time'}
        delay_concepts = {'delay', 'late', 'slow', 'holdup', 'queue'}
        route_concepts = {'route', 'way', 'path', 'directions'}

        has_traffic_concept = bool(traffic_concepts & expanded_words)
        has_greeting_concept = bool(greeting_concepts & expanded_words)
        has_goodbye_concept = bool(goodbye_concepts & expanded_words)
        has_disruption_concept = bool(disruption_concepts & expanded_words)
        has_transit_concept = bool(transit_concepts & expanded_words)
        has_timetable_concept = bool(timetable_concepts & expanded_words)
        has_delay_concept = bool(delay_concepts & expanded_words)
        has_route_concept = bool(route_concepts & expanded_words)
        print(
            "[NLP] Concepts - "
            f"traffic={has_traffic_concept}, greeting={has_greeting_concept}, "
            f"goodbye={has_goodbye_concept}, disruption={has_disruption_concept}, "
            f"transit={has_transit_concept}, timetable={has_timetable_concept}, "
            f"delay={has_delay_concept}, route={has_route_concept}"
        )
        
        # Greeting patterns
        if has_greeting_concept or any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting', 0.9
        
        # Goodbye patterns
        if has_goodbye_concept or any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return 'goodbye', 0.9
        
        # G) Public transport disruption + multimodal
        if (
            has_disruption_concept and has_transit_concept
        ) or any(
            phrase in text_lower
            for phrase in [
                'tube disruption', 'line disruption', 'train disruption',
                'transit disruption', 'bus disruption', 'disruption on the bus',
                'disruption on bus', 'disruption on the train', 'disruption on train',
                'disruption on the tube', 'disruption on tube', 'public transport'
            ]
        ):
            return 'ask_transit_disruption', 0.85
        if any(phrase in text_lower for phrase in ['faster than driving', 'public transport faster', 'multimodal', 'park and ride']):
            return 'ask_multimodal', 0.85

        # G2) Transit timetable/times - check early before generic traffic queries
        if has_timetable_concept and has_transit_concept or any(
            phrase in text_lower
            for phrase in [
                'bus times', 'train times', 'tube times', 'bus or train times',
                'train or bus times', 'timetable', 'next bus', 'next train', 'when is the next'
            ]
        ):
            return 'ask_timetable', 0.9
        
        # C) ETA / arrival time (delay queries only)
        if has_delay_concept or any(
            phrase in text_lower
            for phrase in ['delay', 'how long', 'wait time', 'stuck', 'slow']
        ):
            return 'ask_delay', 0.85
        
        # A) Current conditions
        # Check for "what is the traffic like" pattern first (before generic "is the")
        if any(
            phrase in text_lower
            for phrase in ['what is the traffic like', 'what\'s the traffic like', 'traffic like in', 'traffic like on']
        ):
            return 'ask_traffic_status', 0.9
        if any(
            phrase in text_lower
            for phrase in ['how\'s traffic', 'traffic right now', 'traffic now', 'traffic status', 'traffic condition', 'moving', 'congestion near']
        ) or has_traffic_concept:
            return 'check_current_conditions', 0.85
        # Generic "is the" pattern
        if 'is the' in text_lower and ('traffic' in text_lower or has_traffic_concept):
            return 'ask_traffic_status', 0.85
        if 'traffic' in text_lower or has_traffic_concept or any(phrase in text_lower for phrase in ['how is traffic']):
            return 'ask_traffic_status', 0.85
        
        # Other patterns (congestion-specific)
        if has_traffic_concept or any(phrase in text_lower for phrase in ['congestion', 'jam', 'busy', 'crowded']):
            return 'ask_congestion', 0.85
        
        return 'unknown', 0.5
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract entities like location, route, time, origin, destination
        
        Uses regex patterns and keyword matching
        """
        entities = {}
        text_lower = text.lower()
        
        # Timetable location ("times for X", "at X") is resolved via CSV fuzzy match in _refine_with_stop_datasets.
        # Extract location from "traffic like in [location]" or "traffic in [location]" patterns
        # This must come before generic location patterns to avoid capturing query words
        if 'location' not in entities:
            traffic_in_pattern = r'traffic\s+(?:like\s+)?in\s+([a-z]+(?:\s+[a-z]+)*)'
            match = re.search(traffic_in_pattern, text_lower)
            if match:
                location_raw = match.group(1).strip()
                # Filter out query words
                query_words = ['is', 'the', 'like', 'traffic', 'what', 'how', 'where']
                words = location_raw.split()
                cleaned_words = [w for w in words if w not in query_words]
                # Validate: must have at least one meaningful word
                if cleaned_words and len(' '.join(cleaned_words).strip()) >= 2:
                    cleaned_location = ' '.join(word.capitalize() for word in cleaned_words)
                    entities['location'] = cleaned_location
        
        # Also check for "traffic on [route]" patterns
        traffic_on_pattern = r'traffic\s+(?:like\s+)?on\s+([a-z]+(?:\s+[a-z]+)*(?:\s+(?:highway|road|street|avenue|way))?)'
        match = re.search(traffic_on_pattern, text_lower)
        if match and 'location' not in entities:
            route_raw = match.group(1).strip()
            query_words = ['is', 'the', 'like', 'traffic', 'what', 'how']
            words = route_raw.split()
            cleaned_words = [w for w in words if w not in query_words]
            # Validate: must have at least one meaningful word
            if cleaned_words and len(' '.join(cleaned_words).strip()) >= 2:
                cleaned_route = ' '.join(word.capitalize() for word in cleaned_words)
                entities['route'] = cleaned_route
        
        # Extract bus route numbers early (before generic location patterns)
        # This handles: "bus 83", "the bus 83", "route 83", "disruption on the bus 83"
        bus_route_patterns = [
            r'(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
            r'bus\s+route\s+(\d{1,3}|[Nn]\d{1,3})',
            # Allow "route 83" at end of phrase or before "bus"
            r'route\s+(\d{1,3}|[Nn]\d{1,3})(?:\s+bus)?\b',
            r'(?:disruption|status|delay)\s+(?:on\s+)?(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
        ]
        for pattern in bus_route_patterns:
            match = re.search(pattern, text_lower)
            if match:
                route_num = match.group(1).upper()
                # Set both route and line for disruption handler
                entities['route'] = route_num
                entities['line'] = route_num
                break
        
        # Extract location/route (common patterns)
        # Look for phrases like "on Highway 101", "in downtown", "to airport"
        # Also recognize London transport lines and stations
        location_patterns = [
            r'(?:on|in|at|near|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Capitalized place names
            r'(?:on|in|at|near|to|from)\s+([a-z]+\s+(?:street|road|avenue|highway|freeway|boulevard|way))',
            r'highway\s+(\d+)',
            r'route\s+(\d+)',
            r'hwy\s+(\d+)',
            # London transport patterns
            r'(?:the\s+)?(bakerloo|central|circle|district|hammersmith|jubilee|metropolitan|northern|piccadilly|victoria|waterloo|dlr|overground|tram)\s+(?:line|tube)',
            r'(?:line|tube)\s+(bakerloo|central|circle|district|hammersmith|jubilee|metropolitan|northern|piccadilly|victoria|waterloo)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                match_value = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
                # Clean up: remove common query words
                query_words = ['is', 'the', 'like', 'traffic', 'what', 'how', 'where']
                words = match_value.split()
                cleaned_words = [w for w in words if w.lower() not in query_words]
                if not cleaned_words:
                    continue
                match_value = ' '.join(cleaned_words)
                
                # Skip if this is "bus" and we're in a disruption query (bus route will be extracted separately)
                if match_value.lower() == 'bus' and ('disruption' in text_lower or 'status' in text_lower):
                    continue
                
                # Check if it's a known train line (from _train_lines)
                match_norm = self._normalize_for_line_match(match_value)
                is_train_line = any(
                    self._normalize_for_line_match(ln) in match_norm or match_norm in self._normalize_for_line_match(ln)
                    for ln in self._train_lines
                )
                
                # Don't extract as route if it's a timetable query - we already have the location
                if ('times' in text_lower or 'timetable' in text_lower) and 'location' in entities:
                    # This is a timetable query, skip route extraction entirely
                    continue
                
                # Don't extract as route if the match contains "for" and we're in a timetable query
                if ('times' in text_lower or 'timetable' in text_lower) and 'for' in match_value.lower():
                    # This is likely "times for [location]", don't extract as route
                    continue
                
                if is_train_line:
                    if 'route' not in entities:
                        entities['route'] = match_value
                else:
                    # Only set location for place/street names; do not set route
                    # (route is set by bus_route_patterns or tube line patterns above)
                    # Skip regex location for timetable-like queries (for/near/from/to etc.) so CSV method captures from full text
                    is_timetable_query = any(
                        phrase in text_lower for phrase in [
                            'times', 'timetable', 'next bus', 'next train', 'next tube',
                            'bus times', 'train times', 'tube times'
                        ]
                    )
                    if 'location' not in entities and not is_timetable_query:
                        entities['location'] = match_value
        
        # Extract time
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(?:am|pm)?',
            r'(\d{1,2})\s*(?:am|pm)',
            r'(morning|afternoon|evening|night|noon|midnight)',
            r'(today|tomorrow|now|later)',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                entities['time'] = match.group(0)
                break
        
        # Extract origin and destination
        # Patterns like "from X to Y", "X to Y"
        origin_dest_pattern = r'(?:from\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:station|st|road|street|avenue|highway))?)\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:station|st|road|street|avenue|highway))?)'
        match = re.search(origin_dest_pattern, text, re.IGNORECASE)
        if match:
            entities['origin'] = match.group(1).strip()
            entities['destination'] = match.group(2).strip()
        
        # Also check for "to X" pattern for destination (skip for timetable queries so CSV captures location)
        to_pattern = r'to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:station|st|road|street|avenue|highway))?)'
        match = re.search(to_pattern, text, re.IGNORECASE)
        if match and 'destination' not in entities:
            is_timetable = any(p in text_lower for p in ['times', 'timetable', 'next bus', 'next train', 'next tube', 'bus times', 'train times', 'tube times'])
            if not is_timetable:
                entities['destination'] = match.group(1).strip()
        
        # Extract travel mode
        if re.search(r'\b(drive|driving|car)\b', text_lower):
            entities['travel_mode'] = 'drive'
        elif re.search(r'\b(transit|train|tube|bus|public transport|transport)\b', text_lower):
            entities['travel_mode'] = 'transit'
        elif re.search(r'\b(bike|bicycle|cycling)\b', text_lower):
            entities['travel_mode'] = 'bike'
        elif re.search(r'\b(walk|walking|on foot)\b', text_lower):
            entities['travel_mode'] = 'walk'
        
        # Extract timetable mode preference (bus, train, or both)
        if any(phrase in text_lower for phrase in ['bus times', 'bus time', 'next bus']):
            if 'train' not in text_lower and 'tube' not in text_lower:
                entities['timetable_mode'] = 'bus'
            else:
                entities['timetable_mode'] = 'both'
        elif any(phrase in text_lower for phrase in ['train times', 'train time', 'tube times', 'tube time', 'next train', 'next tube']):
            if 'bus' not in text_lower:
                entities['timetable_mode'] = 'train'
            else:
                entities['timetable_mode'] = 'both'
        elif any(phrase in text_lower for phrase in ['bus or train times', 'train or bus times', 'bus and train times', 'timetable']):
            entities['timetable_mode'] = 'both'
        
        # Extract preferences
        if re.search(r'\b(avoid tolls|no tolls|without tolls)\b', text_lower):
            entities['avoid_tolls'] = True
        if re.search(r'\b(avoid motorways|no motorways|without motorways)\b', text_lower):
            entities['avoid_motorways'] = True
        if re.search(r'\b(avoid ferries|no ferries)\b', text_lower):
            entities['avoid_ferries'] = True
        
        # Extract ULEZ/LEZ/Congestion charge queries
        if re.search(r'\b(ulez|lez|congestion charge)\b', text_lower):
            entities['ulez'] = True
            entities['congestion_charge'] = True
        
        # Extract avoid area
        avoid_pattern = r'avoid\s+(?:this\s+)?area[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(avoid_pattern, text, re.IGNORECASE)
        if match:
            entities['avoid_area'] = match.group(1).strip()
        
        # Extract event information
        event_pattern = r'(?:going to|match at|event at|concert at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(event_pattern, text, re.IGNORECASE)
        if match:
            entities['event'] = match.group(1).strip()
        
        # Extract line/route for transit
        line_pattern = r'(?:the\s+)?(central\s+line|bakerloo\s+line|northern\s+line|piccadilly\s+line|etc\.?)'
        match = re.search(line_pattern, text_lower)
        if match:
            entities['line'] = match.group(1).strip()
        
        # Extract road names for incidents
        road_pattern = r'\b(A\d+|M\d+|M\d+[A-Z]|Junction\s+\d+)\b'
        match = re.search(road_pattern, text, re.IGNORECASE)
        if match:
            entities['road'] = match.group(1).strip()
        
        # Extract common location keywords
        common_locations = {
            'downtown': 'downtown',
            'airport': 'airport',
            'city center': 'city center',
            'suburb': 'suburbs',
            'highway': 'highway',
            'freeway': 'freeway',
            # London stations
            'oxford circus': 'Oxford Circus',
            'kings cross': "King's Cross",
            'paddington': 'Paddington',
            'victoria': 'Victoria',
            'waterloo': 'Waterloo',
            'liverpool street': 'Liverpool Street',
            'euston': 'Euston',
            'st pancras': "St Pancras",
            'charing cross': 'Charing Cross',
            'piccadilly circus': 'Piccadilly Circus',
        }
        
        for keyword, location in common_locations.items():
            if keyword in text_lower and 'location' not in entities:
                entities['location'] = location
                break
        
        return entities
    
    def _load_stop_datasets(self) -> None:
        """
        Load bus and train stop names from local CSVs for fuzzy NER-style matching.
        """
        if self._stops_loaded:
            return


        self._bus_stops = []
        self._train_stations = []
        self._train_lines = []
        self._bus_routes = set()
        self._stops_loaded = True
    
    # Keywords that indicate a status/disruption query (train or bus)
    _DISRUPTION_KEYWORDS = (
        'status', 'disruption', 'disrupted', 'delay', 'delays', 'closure', 'closed',
        'problem', 'problems', 'issue', 'issues', 'service', 'running', 'working'
    )
    
    def _normalize_for_line_match(self, text: str) -> str:
        """Normalize text for train line matching: lowercase, & interchangeable with 'and', apostrophes optional."""
        t = text.lower().strip()
        t = re.sub(r"['\u2019]", '', t)  # remove apostrophes
        t = re.sub(r'\s*&\s*', ' and ', t)
        t = re.sub(r'\s+', ' ', t)
        return t
    
    def extract_train_disruption_line(self, query: str) -> Optional[str]:
        """
        If the query is about train/tube/Overground/DLR status or disruption and mentions
        a line from _train_lines, return that line's display name. Otherwise return None.
        Matching: punctuation like & is interchangeable; apostrophes don't have to be included.
        """
        return None
    
    def extract_bus_disruption_route(self, query: str) -> Optional[str]:
        """
        If the query is about bus status or disruption and contains a route number/string
        that appears in tfl_bus_routes.txt, return that route id (e.g. "83", "N29"). Otherwise return None.
        """
        return None
    
    def parse_line_or_route_followup(self, message: str) -> Optional[Tuple[str, str]]:
        """
        For follow-up replies after "couldn't find a train/bus": if the message is just a train line
        or bus route (no status/disruption keywords required), return (value, 'train') or (value, 'bus').
        Otherwise return None. Bus route is preferred when the message is only digits or N+digits.
        """
        return None
    
    def _best_csv_stop_match(self, candidate: str, mode_hint: Optional[str] = None):
        """
        Given a free-text candidate (usually a location phrase), find the best
        fuzzy match across bus and/or train CSV stop names.
        Returns a dict with keys: type ('bus'|'train'), name, score, or None.

        mode_hint: If "train", only match against train stations (so train
        queries don't match bus CSV). If "bus", only match against bus stops.
        If None, search both and return the single best score (previous behaviour).
        """
        if not candidate or not candidate.strip():
            return None
        
        if not self._stops_loaded:
            self._load_stop_datasets()
        
        cand = candidate.strip().lower()
        if not cand:
            return None
        # Avoid mapping very short, single-word phrases like "home" or "uni"
        # directly to a random stop via fuzzy matching. These are typically
        # handled via user-defined shortcuts upstream; if no shortcut was
        # applied, it's safer to leave them as-is.
        cand_tokens = cand.split()
        if len(cand_tokens) == 1 and len(cand_tokens[0]) <= 4:
            return None
        
        best_type = None
        best_name = None
        best_score = 0.0
        
        # Helper: score a stop name against the candidate, preferring
        # exact substring containment over general fuzziness.
        def _score_stop(stop_name: str) -> float:
            stop_lower = stop_name.lower()
            # If the full stop name appears inside the candidate text,
            # treat this as a near-perfect match even if there are extra words.
            if stop_lower and stop_lower in cand:
                return 1.0
            # If the candidate appears inside the stop name (short query like "Baker"),
            # still give a strong score.
            if cand and cand in stop_lower:
                return 0.95
            # Fallback to character-based similarity for typos / small deviations.
            return SequenceMatcher(None, cand, stop_lower).ratio()
        
        search_bus = mode_hint != "train"
        search_train = mode_hint != "bus"

        if search_bus:
            for name in self._bus_stops:
                s = _score_stop(name)
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_type = "bus"

        if search_train:
            for name in self._train_stations:
                s = _score_stop(name)
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_type = "train"
        
        if best_name is None or best_type is None:
            return None
        
        return {"type": best_type, "name": best_name, "score": best_score}
    
    def _refine_with_stop_datasets(
        self,
        original_text: str,
        intent: str,
        entities: Dict[str, Any],
        confidence: float,
    ):
        """
        Use CSV stop names as an additional NER + slot-filling and intent hint layer.
        """
        return intent, entities, confidence
