from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import os
import re

from chatbot import TrafficChatbot
from journey_planner import JourneyChatbot

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Very simple in-memory user store for demo purposes only.
# In production you would use a database and password hashing.
USERS = {}

# Initialize chatbots
traffic_chatbot = TrafficChatbot()
journey_chatbot = JourneyChatbot()


@app.route('/')
def index():
    """Main page with chat + journey planner interface"""
    # Google Places key is optional; the frontend will still work without it.
    return render_template(
        'index.html',
        google_places_api_key=os.getenv('GOOGLE_PLACES_API_KEY', ''),
    )


@app.route('/login', methods=['GET'])
def login_page():
    """Dedicated login / sign-up page."""
    return render_template('login.html')


@app.route('/me', methods=['GET'])
def me():
    """Return the currently logged-in user (if any)."""
    username = session.get('username')
    return jsonify({'username': username})


def _current_user_key() -> str:
    """Return the key used for per-user in-memory state."""
    return session.get('username') or '_anon'


@app.route('/login', methods=['POST'])
def login():
    """Login or create account. New accounts: username > 3 chars, password >= 8 chars."""
    data = request.get_json(force=True) or {}
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password are required.'}), 400

    is_new_user = username not in USERS
    if is_new_user:
        if len(username) <= 3:
            return jsonify({'error': 'Username must be longer than 3 characters.'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters.'}), 400
    else:
        if USERS[username] != password:
            return jsonify({'error': 'Incorrect password.'}), 400

    USERS[username] = password
    session['username'] = username
    return jsonify({'username': username})


@app.route('/logout', methods=['POST'])
def logout():
    """Log out the current user."""
    session.pop('username', None)
    return jsonify({'ok': True})


def _looks_like_journey_message(text: str) -> bool:
    """
    Heuristic: decide if a free-text message is asking to plan a journey.
    This lets us route to the JourneyChatbot when appropriate while
    keeping the existing traffic assistant for everything else.
    """
    t = (text or '').lower()
    if ' from ' in t and ' to ' in t:
        return True
    if 'plan a journey' in t or 'plan journey' in t or 'get to' in t:
        return True
    return False


@app.route('/chat', methods=['POST'])
def chat():
    """Unified chat endpoint for both the traffic assistant and journey planner."""
    try:
        data = request.get_json(force=True) or {}
        user_message_raw = (data.get('message') or '').strip()

        # Structured journey planner fields (from the journey planner UI)
        from_text_raw = (data.get('from') or '').strip()
        to_text_raw = (data.get('to') or '').strip()
        from_id = (data.get('fromId') or '').strip() or None
        to_id = (data.get('toId') or '').strip() or None
        date_str = (data.get('date') or '').strip() or None
        time_str = (data.get('time') or '').strip() or None


        user_key = _current_user_key()
        user_message = user_message_raw
        from_text = from_text_raw
        to_text = to_text_raw

        now = datetime.utcnow()

        username = session.get('username')

        # 1) If we have structured journey inputs, always use the journey planner.
        if from_text and to_text:
            jp_response = journey_chatbot.handle_structured_journey(
                from_text=from_text,
                to_text=to_text,
                from_id=from_id,
                to_id=to_id,
                date_str=date_str,
                time_str=time_str,
                now=now,
                username=username,
            )

            return jsonify({
                # Main text used by existing frontend
                'response': jp_response.get('reply', ''),
                # Journey-specific payload for the planner UI
                'journeys': jp_response.get('journeys', []),
                'tfl_journey_url': jp_response.get('tfl_journey_url'),
                'disambiguation': jp_response.get('disambiguation', False),
                'from_id': from_id or '',
                'to_id': to_id or '',
                # Basic intent/metadata so the existing info panel still works
                'intent': 'journey_planner',
                'entities': {
                    'from': from_text,
                    'to': to_text,
                },
                'confidence': 1.0,
            })

        # 2) If the free-text clearly looks like a journey query, use the journey planner.
        if _looks_like_journey_message(user_message):
            jp_response = journey_chatbot.handle_message(user_message, now=now, username=username)
            return jsonify({
                'response': jp_response.get('reply', ''),
                'journeys': jp_response.get('journeys', []),
                'tfl_journey_url': jp_response.get('tfl_journey_url'),
                'disambiguation': jp_response.get('disambiguation', False),
                'intent': 'journey_planner',
                'entities': {},
                'confidence': 0.95,
            })

        # 3) Fallback: use the existing traffic chatbot for everything else.
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        traffic_response = traffic_chatbot.process_message(
            user_message, user_key=_current_user_key(), username=username
        )

        return jsonify({
            'response': traffic_response['message'],
            'intent': traffic_response.get('intent', 'unknown'),
            'entities': traffic_response.get('entities', {}),
            'confidence': traffic_response.get('confidence', 0.0),
            'journeys': [],
            'disambiguation': False,
            'timetable': traffic_response.get('timetable'),
            'disruption': traffic_response.get('disruption'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500





@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Travel assistant'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
