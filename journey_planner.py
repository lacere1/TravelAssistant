from datetime import datetime


class JourneyChatbot:
    def __init__(self):
        pass

    def handle_structured_journey(self, from_text, to_text, now=None, username=None, from_id=None, to_id=None, date_str=None, time_str=None):
        if not now:
            now = datetime.utcnow()

        return {
            'reply': f'Planning journey from {from_text} to {to_text}.',
            'journeys': []
        }

    def handle_message(self, user_message, now=None, username=None):
        if not now:
            now = datetime.utcnow()

        return {
            'reply': f'I can help you plan a journey. Please provide start and destination locations.',
            'journeys': []
        }
