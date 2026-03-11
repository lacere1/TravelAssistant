"""
Smart Traffic Query Assistant - Main Chatbot Core
Handles NLP processing, traffic data fetching, and ML predictions
"""
from nlp_processor import NLPProcessor
from transport_api import TransportDataFetcher
import json
import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Any, List, Tuple

class TrafficChatbot:
    def __init__(self):
        """Initialize chatbot with NLP, API, and ML components"""
        self.nlp = NLPProcessor()
        self.transport_api = TransportDataFetcher()
        self.conversation_state: Dict[str, Any] = {
            'origin': None,
            'destination': None,
            'timetable_disambiguation': None,
            'awaiting_disruption_line': None,
        }
        self._user_preferences: Dict[str, Dict[str, Any]] = {}
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
            user_key: Optional per-user/session key for preferences
            username: Optional display name for personalised replies

        Returns:
            dict: Response with message, intent, entities, and confidence
        """
        nlp_result = self.nlp.process(user_message)
        intent = nlp_result['intent']
        entities = nlp_result['entities']
        confidence = nlp_result['confidence']

        self._update_conversation_state(entities, intent)

        response_message = self._generate_response(
            intent, entities, user_message, user_key=user_key, username=username
        )

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
        text_norm = ' ' + re.sub(r'\s+', ' ', text) + ' '

        for opt in options:
            platform = (opt.get('platform') or '').strip().upper()
            if not platform:
                continue
            pl = platform.lower()
            if re.search(r'\bstop\s+' + re.escape(pl) + r'\b', text_norm):
                return opt
            if len(pl) >= 2 and re.search(r'\b' + re.escape(pl) + r'\b', text_norm):
                return opt
            if len(pl) == 1 and re.search(r'\bstop\s+' + re.escape(pl) + r'\b', text_norm):
                return opt

        for opt in options:
            platform = (opt.get('platform') or '').strip()
            if not platform or not platform.isdigit():
                continue
            if re.search(r'\bplatform\s*' + re.escape(platform) + r'\b', text_norm):
                return opt
            if re.search(r'\b' + re.escape(platform) + r'\b', text_norm) and re.search(r'\bplatform\b', text_norm):
                return opt

        for opt in options:
            towards = (opt.get('towards') or '').strip().lower()
            if not towards:
                continue
            if towards in text:
                return opt
            for word in towards.split():
                if len(word) >= 4 and word in text:
                    return opt

        for opt in options:
            direction = (opt.get('direction') or opt.get('name') or opt.get('id') or '').strip().lower()
            if not direction:
                continue
            if direction in text:
                return opt
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

        for opt in options:
            name = (opt.get('label') or opt.get('name') or '').strip().lower()
            if not name:
                continue
            if name in text or text in name:
                return opt
            name_words = [w for w in name.split() if len(w) >= 3 and w not in ('station', 'underground')]
            if name_words and all(w in text for w in name_words):
                return opt

        return None

    def _classify_disambiguation_reply(
        self, user_message: str, options: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]], float]:
        """
        Classify user response to a bus stop disambiguation prompt.
        """
        if not options:
            return 'unknown', None, 0.5

        matched_opt = self._resolve_disambiguation_reply(user_message, options)
        if matched_opt:
            return 'bus_stop_selected', matched_opt, 0.95

        return 'bus_stop_disambiguation_unclear', None, 0.6

    def _generate_response(
        self, intent: str, entities: Dict[str, str], user_message: str, user_key: Optional[str] = None, username: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response based on intent"""

        if intent == 'greeting':
            msg = "Hello!"
            if username:
                msg += f" How can I help you, {username}?"
            else:
                msg += " How can I help you?"
            return {'message': msg}

        if intent == 'goodbye':
            msg = "Goodbye!"
            if username:
                msg += f" Safe travels, {username}!"
            return {'message': msg}

        if intent == 'ask_traffic_status':
            location = entities.get('location', 'your area')
            msg = f"I can check traffic information for {location}."
            return {'message': msg}

        if intent == 'ask_delay':
            msg = "Checking delay information..."
            return {'message': msg}

        if intent == 'ask_timetable':
            location = entities.get('location', 'nearby')
            timetable_mode = entities.get('timetable_mode', 'transit')
            msg = f"Here are {timetable_mode} times for {location}."
            return {'message': msg}

        msg = "I can help with traffic information and journey planning."
        return {'message': msg}

    def _format_response(
        self, response_message: Dict[str, Any], intent: str, entities: Dict[str, str]
    ) -> str:
        """Format response for display"""
        return response_message.get('message', '')
