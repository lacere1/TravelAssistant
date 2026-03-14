"""
NLP Processor (60% Prototype)
Handles intent detection and entity extraction
"""
import re
import os
from difflib import SequenceMatcher
from typing import Dict, List, Any, Set, Optional, Tuple


class NLPProcessor:
    def __init__(self):
        print("Loading NLP models...")
        self.intent_classifier = None
        
        self.intent_labels = [
            "check_current_conditions", "ask_traffic_status", "ask_delay",
            "ask_transit_disruption", "ask_multimodal", "ask_timetable",
            "ask_transit_times", "ask_congestion", "greeting", "goodbye"
        ]
        
        print("NLP models loaded successfully")
        
        self._bus_stops: List[str] = []
        self._train_stations: List[str] = []
        self._train_lines: List[str] = []
        self._bus_routes: Set[str] = set()
        self._stops_loaded: bool = False
        self._load_stop_datasets()
    
    def process(self, text: str) -> Dict[str, Any]:
        original_text = text
        text = text.strip().lower()
        
        intent, confidence = self._classify_intent(text)
        entities = self._extract_entities(text)
        intent, entities, confidence = self._refine_with_stop_datasets(
            original_text, intent, entities, confidence
        )
        
        return {'intent': intent, 'entities': entities, 'confidence': confidence}
    
    def _classify_intent(self, text: str) -> tuple:
        text_lower = text.lower()
        
        # Timetable queries first
        if any(phrase in text_lower for phrase in ['bus times', 'train times', 'tube times', 'bus or train times', 'train or bus times', 'timetable', 'next bus', 'next train', 'when is the next']):
            return 'ask_timetable', 0.9
        
        # Disruption/status
        if any(kw in text_lower for kw in ['disruption', 'disrupted', 'status', 'delay', 'delays']):
            has_train = any(x in text_lower for x in [
                ' line', ' tube', 'train', 'overground', 'dlr', 'underground',
                'bakerloo', 'central', 'circle', 'district', 'hammersmith', 'jubilee',
                'metropolitan', 'northern', 'piccadilly', 'victoria', 'waterloo',
                'windrush', 'lioness', 'mildmay', 'suffragette', 'weaver', 'liberty'
            ])
            has_bus_route = bool(re.search(r'bus.*\d|\d.*bus', text_lower))
            has_bus_status = 'bus' in text_lower
            if has_train or has_bus_route or has_bus_status:
                return 'ask_transit_disruption', 0.9
        
        # Traffic patterns
        if any(phrase in text_lower for phrase in ['what is the traffic like', "what's the traffic like", 'how is the traffic', 'traffic like']):
            return 'ask_traffic_status', 0.9
        if re.search(r'traffic\s+in\s+[a-z]+(?:\s+[a-z]+)*', text_lower):
            return 'ask_traffic_status', 0.9
        
        # Fallback to rule-based
        return self._rule_based_intent(text)
    
    def _rule_based_intent(self, text: str) -> tuple:
        text_lower = text.lower()
        tokens = re.findall(r'\w+', text_lower)
        expanded_words = set(t.lower() for t in tokens if t)

        # Concept sets for matching
        traffic_concepts = {'traffic', 'congestion', 'jam'}
        greeting_concepts = {'hello', 'hi', 'hey', 'greetings'}
        goodbye_concepts = {'bye', 'goodbye', 'farewell'}
        disruption_concepts = {'disruption', 'disturbance', 'interruption'}
        transit_concepts = {'transit', 'train', 'tube', 'subway', 'metro', 'tram', 'bus'}
        timetable_concepts = {'timetable', 'schedule', 'times', 'time'}
        delay_concepts = {'delay', 'late', 'slow', 'holdup', 'queue'}

        has_traffic = bool(traffic_concepts & expanded_words)
        has_greeting = bool(greeting_concepts & expanded_words)
        has_goodbye = bool(goodbye_concepts & expanded_words)
        has_disruption = bool(disruption_concepts & expanded_words)
        has_transit = bool(transit_concepts & expanded_words)
        has_timetable = bool(timetable_concepts & expanded_words)
        has_delay = bool(delay_concepts & expanded_words)
        
        if has_greeting or any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting', 0.9
        if has_goodbye or any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return 'goodbye', 0.9
        
        if (has_disruption and has_transit) or any(
            phrase in text_lower for phrase in [
                'tube disruption', 'line disruption', 'train disruption',
                'transit disruption', 'bus disruption', 'public transport'
            ]
        ):
            return 'ask_transit_disruption', 0.85
        if any(phrase in text_lower for phrase in ['multimodal', 'park and ride']):
            return 'ask_multimodal', 0.85

        if has_timetable and has_transit or any(
            phrase in text_lower for phrase in [
                'bus times', 'train times', 'tube times', 'timetable', 'next bus', 'next train'
            ]
        ):
            return 'ask_timetable', 0.9
        
        if has_delay or any(phrase in text_lower for phrase in ['delay', 'how long', 'wait time', 'stuck', 'slow']):
            return 'ask_delay', 0.85
        
        if any(phrase in text_lower for phrase in ['what is the traffic like', "what's the traffic like"]):
            return 'ask_traffic_status', 0.9
        if any(phrase in text_lower for phrase in ["how's traffic", 'traffic right now', 'traffic now', 'traffic status', 'congestion near']) or has_traffic:
            return 'check_current_conditions', 0.85
        if 'traffic' in text_lower or has_traffic:
            return 'ask_traffic_status', 0.85
        if has_traffic or any(phrase in text_lower for phrase in ['congestion', 'jam', 'busy', 'crowded']):
            return 'ask_congestion', 0.85
        
        return 'unknown', 0.5
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        entities = {}
        text_lower = text.lower()
        
        # Traffic in location
        if 'location' not in entities:
            traffic_in_pattern = r'traffic\s+(?:like\s+)?in\s+([a-z]+(?:\s+[a-z]+)*)'
            match = re.search(traffic_in_pattern, text_lower)
            if match:
                location_raw = match.group(1).strip()
                query_words = ['is', 'the', 'like', 'traffic', 'what', 'how', 'where']
                words = location_raw.split()
                cleaned_words = [w for w in words if w not in query_words]
                if cleaned_words and len(' '.join(cleaned_words).strip()) >= 2:
                    entities['location'] = ' '.join(word.capitalize() for word in cleaned_words)
        
        # Traffic on route
        traffic_on_pattern = r'traffic\s+(?:like\s+)?on\s+([a-z]+(?:\s+[a-z]+)*(?:\s+(?:highway|road|street|avenue|way))?)'
        match = re.search(traffic_on_pattern, text_lower)
        if match and 'location' not in entities:
            route_raw = match.group(1).strip()
            query_words = ['is', 'the', 'like', 'traffic', 'what', 'how']
            words = route_raw.split()
            cleaned_words = [w for w in words if w not in query_words]
            if cleaned_words and len(' '.join(cleaned_words).strip()) >= 2:
                entities['route'] = ' '.join(word.capitalize() for word in cleaned_words)
        
        # Bus route numbers
        bus_route_patterns = [
            r'(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
            r'bus\s+route\s+(\d{1,3}|[Nn]\d{1,3})',
            r'route\s+(\d{1,3}|[Nn]\d{1,3})(?:\s+bus)?\b',
            r'(?:disruption|status|delay)\s+(?:on\s+)?(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
        ]
        for pattern in bus_route_patterns:
            match = re.search(pattern, text_lower)
            if match:
                route_num = match.group(1).upper()
                entities['route'] = route_num
                entities['line'] = route_num
                break
        
        # Location patterns
        location_patterns = [
            r'(?:on|in|at|near|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:on|in|at|near|to|from)\s+([a-z]+\s+(?:street|road|avenue|highway|freeway|boulevard|way))',
            r'highway\s+(\d+)',
            r'route\s+(\d+)',
            r'(?:the\s+)?(bakerloo|central|circle|district|hammersmith|jubilee|metropolitan|northern|piccadilly|victoria|waterloo|dlr|overground|tram)\s+(?:line|tube)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                match_value = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
                query_words = ['is', 'the', 'like', 'traffic', 'what', 'how', 'where']
                words = match_value.split()
                cleaned_words = [w for w in words if w.lower() not in query_words]
                if not cleaned_words:
                    continue
                match_value = ' '.join(cleaned_words)
                
                if match_value.lower() == 'bus' and ('disruption' in text_lower or 'status' in text_lower):
                    continue
                
                match_norm = self._normalize_for_line_match(match_value)
                is_train_line = any(
                    self._normalize_for_line_match(ln) in match_norm or match_norm in self._normalize_for_line_match(ln)
                    for ln in self._train_lines
                )
                
                if ('times' in text_lower or 'timetable' in text_lower) and 'location' in entities:
                    continue
                
                if is_train_line:
                    if 'route' not in entities:
                        entities['route'] = match_value
                else:
                    is_timetable_query = any(
                        phrase in text_lower for phrase in ['times', 'timetable', 'next bus', 'next train', 'next tube']
                    )
                    if 'location' not in entities and not is_timetable_query:
                        entities['location'] = match_value
        
        # Time
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
        
        # Origin and destination
        origin_dest_pattern = r'(?:from\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:station|st|road|street|avenue|highway))?)\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:station|st|road|street|avenue|highway))?)'
        match = re.search(origin_dest_pattern, text, re.IGNORECASE)
        if match:
            entities['origin'] = match.group(1).strip()
            entities['destination'] = match.group(2).strip()
        
        # Travel mode
        if re.search(r'\b(drive|driving|car)\b', text_lower):
            entities['travel_mode'] = 'drive'
        elif re.search(r'\b(transit|train|tube|bus|public transport|transport)\b', text_lower):
            entities['travel_mode'] = 'transit'
        elif re.search(r'\b(bike|bicycle|cycling)\b', text_lower):
            entities['travel_mode'] = 'bike'
        elif re.search(r'\b(walk|walking|on foot)\b', text_lower):
            entities['travel_mode'] = 'walk'
        
        # Timetable mode
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
        elif any(phrase in text_lower for phrase in ['bus or train times', 'train or bus times', 'timetable']):
            entities['timetable_mode'] = 'both'
        
        # Preferences
        if re.search(r'\b(avoid tolls|no tolls|without tolls)\b', text_lower):
            entities['avoid_tolls'] = True
        if re.search(r'\b(avoid motorways|no motorways|without motorways)\b', text_lower):
            entities['avoid_motorways'] = True
        
        # Common locations
        common_locations = {
            'oxford circus': 'Oxford Circus', 'kings cross': "King's Cross",
            'paddington': 'Paddington', 'victoria': 'Victoria', 'waterloo': 'Waterloo',
            'liverpool street': 'Liverpool Street', 'euston': 'Euston',
            'st pancras': "St Pancras", 'charing cross': 'Charing Cross',
            'piccadilly circus': 'Piccadilly Circus',
        }
        for keyword, location in common_locations.items():
            if keyword in text_lower and 'location' not in entities:
                entities['location'] = location
                break
        
        return entities
    
    def _load_stop_datasets(self) -> None:
        if self._stops_loaded:
            return
        self._bus_stops = []
        self._train_stations = []
        self._train_lines = []
        self._bus_routes = set()
        self._stops_loaded = True
    
    _DISRUPTION_KEYWORDS = (
        'status', 'disruption', 'disrupted', 'delay', 'delays', 'closure', 'closed',
        'problem', 'problems', 'issue', 'issues', 'service', 'running', 'working'
    )
    
    def _normalize_for_line_match(self, text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"['\u2019]", '', t)
        t = re.sub(r'\s*&\s*', ' and ', t)
        t = re.sub(r'\s+', ' ', t)
        return t
    
    def extract_train_disruption_line(self, query: str) -> Optional[str]:
        return None
    
    def extract_bus_disruption_route(self, query: str) -> Optional[str]:
        return None
    
    def parse_line_or_route_followup(self, message: str) -> Optional[Tuple[str, str]]:
        return None
    
    def _best_csv_stop_match(self, candidate: str, mode_hint: Optional[str] = None):
        if not candidate or not candidate.strip():
            return None
        if not self._stops_loaded:
            self._load_stop_datasets()
        cand = candidate.strip().lower()
        if not cand:
            return None
        cand_tokens = cand.split()
        if len(cand_tokens) == 1 and len(cand_tokens[0]) <= 4:
            return None
        return None
    
    def _refine_with_stop_datasets(self, original_text: str, intent: str, entities: Dict[str, Any], confidence: float):
        return intent, entities, confidence
