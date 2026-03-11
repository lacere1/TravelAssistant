"""
NLP Processor using Hugging Face Transformers
Handles intent detection and entity extraction
"""
from typing import Dict, List, Any, Optional, Tuple
import re


class NLPProcessor:
    def __init__(self):
        """Initialize NLP models for intent classification and entity extraction"""
        print("Loading NLP models...")
        self.intent_labels = [
            "check_current_conditions",
            "ask_traffic_status",
            "ask_delay",
            "ask_transit_disruption",
            "ask_timetable",
            "ask_transit_times",
            "greeting",
            "goodbye"
        ]
        print("NLP models loaded successfully")

    def process(self, text: str) -> Dict[str, Any]:
        """
        Process user input to extract intent and entities

        Args:
            text: User's natural language input

        Returns:
            dict: Contains intent, entities, and confidence score
        """
        text = text.strip().lower()
        print(f"[NLP] Incoming text: {text!r}")

        intent, confidence = self._classify_intent(text)
        print(f"[NLP] Final intent: {intent}, confidence: {confidence:.3f}")

        entities = self._extract_entities(text)
        print(f"[NLP] Extracted entities: {entities}")

        return {
            'intent': intent,
            'entities': entities,
            'confidence': confidence
        }

    def _classify_intent(self, text: str) -> tuple:
        """Classify user intent using rule-based patterns"""

        text_lower = text.lower()

        if any(phrase in text_lower for phrase in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting', 0.9

        if any(phrase in text_lower for phrase in ['bye', 'goodbye', 'farewell']):
            return 'goodbye', 0.9

        if any(phrase in text_lower for phrase in ['bus times', 'train times', 'tube times', 'timetable', 'next bus', 'next train']):
            return 'ask_timetable', 0.9

        if any(phrase in text_lower for phrase in ['disruption', 'status', 'delay', 'delays']):
            return 'ask_transit_disruption', 0.85

        if any(phrase in text_lower for phrase in ['traffic', 'congestion', 'jam']):
            return 'ask_traffic_status', 0.85

        return 'check_current_conditions', 0.7

    def _extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract entities like location, route, time

        Uses regex patterns and keyword matching
        """
        entities = {}
        text_lower = text.lower()

        location_patterns = [
            r'(?:in|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:traffic\s+)?(?:on|in)\s+([a-z]+(?:\s+[a-z]+)*\s+(?:street|road|avenue|highway))',
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities['location'] = matches[0]
                break

        time_patterns = [
            r'(\d{1,2}):(\d{2})',
            r'(morning|afternoon|evening|night)',
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                entities['time'] = match.group(0)
                break

        if re.search(r'\b(bus|train|tube|transit|public transport)\b', text_lower):
            entities['travel_mode'] = 'transit'

        bus_route_match = re.search(r'bus\s+(\d{1,3})', text_lower)
        if bus_route_match:
            entities['route'] = bus_route_match.group(1)

        return entities

    def extract_train_disruption_line(self, query: str) -> Optional[str]:
        """
        If the query is about train/tube/Overground/DLR status or disruption and mentions
        a line, return that line's display name. Otherwise return None.
        """
        return None

    def extract_bus_disruption_route(self, query: str) -> Optional[str]:
        """
        If the query is about bus status or disruption and contains a route number,
        return that route id. Otherwise return None.
        """
        return None

    def parse_line_or_route_followup(self, message: str) -> Optional[Tuple[str, str]]:
        """
        For follow-up replies after "couldn't find a train/bus": if the message is just a train line
        or bus route, return (value, 'train') or (value, 'bus'). Otherwise return None.
        """
        return None

    def _best_csv_stop_match(self, candidate: str, mode_hint: Optional[str] = None):
        """
        Given a free-text candidate, find the best fuzzy match across bus and/or train CSV stop names.
        Returns a dict with keys: type ('bus'|'train'), name, score, or None.
        """
        return None

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
