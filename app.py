from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import os
import re
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
load_dotenv()

from chatbot import TrafficChatbot
from journey_planner import JourneyChatbot, TflJourneyClient

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Very simple in-memory user store for demo purposes only.
# In production you would use a database and password hashing.
USERS = {}

# Very simple in-memory store for user-defined text shortcuts.
# In production this should live in persistent storage.
USER_SHORTCUTS = {}

# Initialize chatbots
traffic_chatbot = TrafficChatbot()
journey_chatbot = JourneyChatbot()
tfl_journey_client = TflJourneyClient()


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


def _apply_shortcuts_to_text(username: str, text: str) -> str:
    """
    Apply user-defined shortcuts to a free-text string.
    Replaces whole-word matches of each shortcut (case-insensitive).
    """
    shortcuts_for_user = USER_SHORTCUTS.get(username) or {}
    if not shortcuts_for_user or not text:
        return text

    result = text
    for raw_key, replacement in shortcuts_for_user.items():
        if not raw_key or not replacement:
            continue
        # \b ensures we only match whole words like "home", not "homework"
        escaped_key = re.escape(raw_key)
        pattern = re.compile(rf"\b{escaped_key}\b", flags=re.IGNORECASE)
        result = pattern.sub(replacement, result)
    return result


@app.route('/shortcuts', methods=['GET', 'POST'])
def shortcuts():
    """
    Simple JSON API for managing user-defined text shortcuts.
    GET  -> list current user's shortcuts
    POST -> create/update a shortcut: { "key": "home", "value": "Neasden Station" }
    """
    username = _current_user_key()

    if request.method == 'GET':
        shortcuts_for_user = USER_SHORTCUTS.get(username) or {}
        return jsonify([
            {'key': k, 'value': v}
            for k, v in sorted(shortcuts_for_user.items())
        ])

    data = request.get_json(force=True) or {}
    key = (data.get('key') or '').strip()
    value = (data.get('value') or '').strip()

    if not key or not value:
        return jsonify({'error': 'Both key and value are required.'}), 400

    shortcuts_for_user = USER_SHORTCUTS.setdefault(username, {})
    shortcuts_for_user[key.lower()] = value

    return jsonify({'ok': True, 'key': key.lower(), 'value': value})


@app.route('/shortcuts/<key>', methods=['DELETE'])
def delete_shortcut(key):
    """Delete a single shortcut for the current user."""
    username = _current_user_key()
    shortcuts_for_user = USER_SHORTCUTS.get(username) or {}
    key_lower = (key or '').lower()

    if key_lower in shortcuts_for_user:
        del shortcuts_for_user[key_lower]
        return jsonify({'ok': True})

    return jsonify({'error': 'Shortcut not found.'}), 404


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


def _journey_planner_entities(state: dict) -> dict:
    """Build entities dict for the info panel from journey planner state."""
    entities = {}
    if not state:
        return entities
    if state.get("fromQuery"):
        entities["from"] = state["fromQuery"]
    if state.get("toQuery"):
        entities["to"] = state["toQuery"]
    when = state.get("when")
    if when and isinstance(when, dict):
        dt = when.get("datetime")
        if hasattr(dt, "strftime"):
            entities["when"] = dt.strftime("%Y-%m-%d %H:%M")
        elif dt is not None:
            entities["when"] = str(dt)
    if state.get("fromLocationId"):
        entities["from_id"] = state["fromLocationId"]
    if state.get("toLocationId"):
        entities["to_id"] = state["toLocationId"]
    return entities


def _looks_like_journey_message(text: str) -> bool:
    """
    Heuristic: decide if a free-text message is asking to plan a journey.
    The word "plan" (e.g. "plan a journey", "plan my journey") triggers
    Journey Planner intent. Also "from X to Y" and "get to" route here.
    """
    t = (text or '').lower()
    if ' from ' in t and ' to ' in t:
        return True
    if 'plan' in t and (
        'journey' in t or 'trip' in t or 'route' in t or ' from ' in t or ' to ' in t or 'get to' in t
    ):
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

        # Apply user-defined shortcuts before routing the message.
        user_key = _current_user_key()
        user_message = _apply_shortcuts_to_text(user_key, user_message_raw)
        from_text = _apply_shortcuts_to_text(user_key, from_text_raw)
        to_text = _apply_shortcuts_to_text(user_key, to_text_raw)

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

            entities = {"from": from_text, "to": to_text}
            if date_str and time_str:
                entities["when"] = f"{date_str} {time_str}"
            if from_id:
                entities["from_id"] = from_id
            if to_id:
                entities["to_id"] = to_id
            return jsonify({
                # Main text used by existing frontend
                "response": jp_response.get("reply", ""),
                # Journey-specific payload for the planner UI
                "journeys": jp_response.get("journeys", []),
                "tfl_journey_url": jp_response.get("tfl_journey_url"),
                "disambiguation": jp_response.get("disambiguation", False),
                "state": jp_response.get("state", {}),
                "from_id": from_id or "",
                "to_id": to_id or "",
                # Basic intent/metadata so the existing info panel still works
                "intent": "journey_planner",
                "entities": entities,
                "confidence": 1.0,
            })

        # 2) If we're in the middle of journey planning (e.g. asked "Where from?", waiting for "3", or for "neasden station"), use the journey planner.
        jp_state = journey_chatbot.state.get("global") or {}
        if jp_state.get("journey_planning_active") or jp_state.get("fromOptions") or jp_state.get("toOptions"):
            jp_response = journey_chatbot.handle_message(user_message, now=now, username=username)
            return jsonify({
                "response": jp_response.get("reply", ""),
                "journeys": jp_response.get("journeys", []),
                "tfl_journey_url": jp_response.get("tfl_journey_url"),
                "disambiguation": jp_response.get("disambiguation", False),
                "journey_disambiguation": jp_response.get("place_disambiguation"),
                "state": jp_response.get("state", {}),
                "intent": "journey_planner",
                "entities": _journey_planner_entities(jp_response.get("state", {})),
                "confidence": 0.95,
            })

        # 3) If the free-text clearly looks like a journey query, use the journey planner.
        if _looks_like_journey_message(user_message):
            jp_response = journey_chatbot.handle_message(user_message, now=now, username=username)
            return jsonify({
                "response": jp_response.get("reply", ""),
                "journeys": jp_response.get("journeys", []),
                "tfl_journey_url": jp_response.get("tfl_journey_url"),
                "disambiguation": jp_response.get("disambiguation", False),
                "journey_disambiguation": jp_response.get("place_disambiguation"),
                "state": jp_response.get("state", {}),
                "intent": "journey_planner",
                "entities": _journey_planner_entities(jp_response.get("state", {})),
                "confidence": 0.95,
            })

        # 4) Fallback: use the existing traffic chatbot for everything else.
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
            'timetable_disambiguation': traffic_response.get('timetable_disambiguation'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/suggest')
def suggest():
    """
    Optional autocomplete endpoint that proxies to TfL's Place Search.
    Currently not used by the main UI, but available for future enhancements.
    """
    query = (request.args.get('q') or '').strip()
    if not query:
        return jsonify([])

    options = tfl_journey_client.search_places(query)
    return jsonify(options)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Travel assistant'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
