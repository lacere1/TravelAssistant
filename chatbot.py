from nlp_processor import NLPProcessor
from transport_api import TransportDataFetcher


class TrafficChatbot:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.transport_api = TransportDataFetcher()

    def process_message(self, user_message, user_key=None, username=None):
        result = self.nlp.process(user_message)
        intent = result.get('intent', 'unknown')
        confidence = result.get('confidence', 0.0)
        entities = result.get('entities', {})

        if intent == 'greeting':
            message = 'Hello! How can I help with your travel?'
        elif intent == 'goodbye':
            message = 'Goodbye! Safe travels!'
        elif intent == 'ask_traffic_status':
            route = entities.get('route', 'your route')
            traffic_data = self.transport_api.get_route_traffic(route)
            if traffic_data:
                message = f'Traffic on {route}: {traffic_data.get("status", "Unknown")}'
            else:
                message = f'Checking traffic conditions on {route}.'
        elif intent == 'ask_delay':
            location = entities.get('location', 'your location')
            message = f'Checking delays around {location}.'
        elif intent == 'ask_congestion':
            location = entities.get('location', 'your route')
            message = f'Checking congestion on {location}.'
        elif intent == 'ask_transit_disruption':
            line = entities.get('line', 'the line')
            disruptions = self.transport_api.get_transit_disruption(line)
            if disruptions:
                message = f'Found disruptions on {line}.'
                return {
                    'message': message,
                    'intent': intent,
                    'entities': entities,
                    'confidence': confidence,
                    'disruption': disruptions
                }
            message = f'Checking disruptions on {line}.'
        elif intent == 'ask_timetable':
            stop = entities.get('stop', 'the stop')
            timetable = self.transport_api.get_tfl_timetable(stop)
            if timetable:
                message = f'Found timetable for {stop}.'
                return {
                    'message': message,
                    'intent': intent,
                    'entities': entities,
                    'confidence': confidence,
                    'timetable': timetable
                }
            message = f'Looking up timetable for {stop}.'
        elif intent == 'ask_bus_disruption':
            route = entities.get('route', 'the bus route')
            disruptions = self.transport_api.get_bus_disruption(route)
            if disruptions:
                message = f'Found bus disruptions on {route}.'
                return {
                    'message': message,
                    'intent': intent,
                    'entities': entities,
                    'confidence': confidence,
                    'disruption': disruptions
                }
            message = f'Checking bus disruptions on {route}.'
        else:
            message = 'I can help with traffic and travel information. Try asking about current conditions.'

        return {
            'message': message,
            'intent': intent,
            'entities': entities,
            'confidence': confidence
        }
