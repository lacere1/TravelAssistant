"""
Smart Traffic Query Assistant - Main Chatbot Core
Handles NLP processing, traffic data fetching, and ML predictions
"""
from nlp_processor import NLPProcessor
from transport_api import TransportDataFetcher
import json
import re
from typing import Dict, Optional, Any

class TrafficChatbot:
    def __init__(self):
        """Initialize chatbot with NLP and API components"""
        self.nlp = NLPProcessor()
        self.transport_api = TransportDataFetcher()
        self.conversation_state: Dict[str, Any] = {
            'origin': None,
            'destination': None,
        }
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
            user_key: Optional per-user/session key
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
            intent, entities, user_message, username=username
        )

        formatted_response = self._format_response(response_message, intent, entities)

        return {
            'message': formatted_response,
            'intent': intent,
            'entities': entities,
            'confidence': confidence,
        }

    def _update_conversation_state(self, entities: Dict[str, str], intent: str):
        """Update conversation state with extracted entities"""
        if entities.get('origin'):
            self.conversation_state['origin'] = entities['origin']
        if entities.get('destination'):
            self.conversation_state['destination'] = entities['destination']

    def _generate_response(
        self, intent: str, entities: Dict[str, str], user_message: str, username: Optional[str] = None
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

        if intent == 'ask_transit_disruption':
            msg = "Checking transit disruption information..."
            return {'message': msg}

        if intent == 'ask_bus_disruption':
            route = entities.get('route', 'the bus route')
            msg = f"Checking disruptions for {route}..."
            return {'message': msg}

        msg = "I can help with traffic information and journey planning."
        return {'message': msg}

    def _format_response(
        self, response_message: Dict[str, Any], intent: str, entities: Dict[str, str]
    ) -> str:
        """Format response for display"""
        return response_message.get('message', '')
