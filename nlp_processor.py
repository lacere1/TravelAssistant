import re


class NLPProcessor:
    def __init__(self):
        self._bus_stops = []
        self._train_stations = []
        self._train_lines = []
        self._bus_routes = set()
        self._stops_loaded = False
        self._load_stop_datasets()

    def _load_stop_datasets(self):
        self._stops_loaded = True

    def _rule_based_intent(self, text):
        text = text.lower()

        if any(word in text for word in ['hello', 'hi', 'hey']):
            return 'greeting', 0.95

        if any(word in text for word in ['goodbye', 'bye']):
            return 'goodbye', 0.95

        if any(word in text for word in ['traffic', 'busy', 'congestion']):
            if any(word in text for word in ['status', 'current', 'how']):
                return 'ask_traffic_status', 0.85

        if any(word in text for word in ['delay', 'late', 'slow']):
            return 'ask_delay', 0.80

        if any(word in text for word in ['congestion', 'congested']):
            return 'ask_congestion', 0.80

        if any(word in text for word in ['disruption', 'closed', 'suspended', 'line']):
            if any(word in text for word in ['train', 'tube', 'underground']):
                return 'ask_transit_disruption', 0.85

        if any(word in text for word in ['bus', 'route']) and any(word in text for word in ['disruption', 'issue']):
            return 'ask_bus_disruption', 0.80

        if any(word in text for word in ['timetable', 'schedule', 'times', 'when']):
            if any(word in text for word in ['train', 'tube', 'bus']):
                return 'ask_timetable', 0.80

        if any(word in text for word in ['timetable', 'schedule', 'times']):
            return 'ask_transit_times', 0.75

        return 'unknown', 0.0

    def _extract_entities(self, text):
        entities = {}
        text_lower = text.lower()

        locations = ['baker street', 'oxford street', 'london', 'piccadilly', 'king cross']
        for loc in locations:
            if loc in text_lower:
                entities['location'] = loc
                break

        routes = ['central', 'northern', 'bakerloo', 'district', 'circle']
        for route in routes:
            if route in text_lower:
                entities['route'] = route
                break

        lines = ['central line', 'northern line', 'bakerloo line', 'circle line']
        for line in lines:
            if line in text_lower:
                entities['line'] = line
                break

        bus_routes = ['15', '73', '205', '139']
        for bus_route in bus_routes:
            if f'route {bus_route}' in text_lower or f'bus {bus_route}' in text_lower:
                entities['bus_route'] = bus_route
                break

        if any(word in text_lower for word in ['departure', 'leave', 'depart']):
            entities['timetable_mode'] = 'departure'
        elif any(word in text_lower for word in ['arrival', 'arrive']):
            entities['timetable_mode'] = 'arrival'

        if any(word in text_lower for word in ['walk', 'walking']):
            entities['travel_mode'] = 'walking'
        elif any(word in text_lower for word in ['bus']):
            entities['travel_mode'] = 'bus'
        elif any(word in text_lower for word in ['tube', 'train', 'underground']):
            entities['travel_mode'] = 'tube'

        stops = ['baker street', 'oxford circus', 'piccadilly circus']
        for stop in stops:
            if stop in text_lower:
                entities['stop'] = stop
                break

        time_match = re.search(r'(\d{1,2}):(\d{2})', text)
        if time_match:
            entities['time'] = time_match.group(0)

        return entities

    def process(self, text):
        intent, confidence = self._rule_based_intent(text)
        entities = self._extract_entities(text)

        return {
            'intent': intent,
            'entities': entities,
            'confidence': confidence
        }
