from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import os
import re
import sqlite3
import json
import sys

# Windows consoles default to cp1252, which cannot encode the arrows and other
# symbols used in this app's diagnostic prints. Without this, a single logging
# line raises UnicodeEncodeError mid-request and turns a working response into
# a 500. Replace rather than fail: logs are diagnostics, not payload.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a reconfigurable TextIO
        pass
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables from the .env file in the project root
load_dotenv()

from chatbot import TrafficChatbot
from journey_planner import JourneyChatbot, TflJourneyClient
# Photo -> location entity, resolved by the LandmarkLens service over HTTP.
import landmark_client

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------
DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'travel_assistant.db')


def get_db() -> sqlite3.Connection:
    """Open a new database connection with Row factory.

    journal_mode=WAL is ideal for concurrent reads, but falls back to DELETE
    on filesystems that don't support POSIX file-locking (e.g. SMB/CIFS mounts).
    """
    conn = sqlite3.connect(DATABASE, timeout=10)
    conn.row_factory = sqlite3.Row
    # Use DELETE journal mode for broadest filesystem compatibility
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username     TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at   TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS shortcuts (
                username TEXT NOT NULL,
                key      TEXT NOT NULL,
                value    TEXT NOT NULL,
                PRIMARY KEY (username, key)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS stop_preferences (
                username             TEXT PRIMARY KEY,
                last_chosen_stop_id  TEXT,
                frequent_stops       TEXT NOT NULL DEFAULT '{}'
            )
        ''')
        # Add journey_history column if it doesn't exist yet (safe migration)
        try:
            conn.execute(
                "ALTER TABLE stop_preferences ADD COLUMN journey_history TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                username TEXT PRIMARY KEY,
                data     TEXT NOT NULL DEFAULT '[]'
            )
        ''')
        conn.commit()


init_db()

# Initialize chatbots
traffic_chatbot = TrafficChatbot()
journey_chatbot = JourneyChatbot()
tfl_journey_client = TflJourneyClient()

# Flask session backup for when the journey planner asked "rephrase this location"
# after OSM returned no results. In-memory global state can be lost on restart;
# session keeps the user in the journey flow until they answer or start a new chat.
JP_REPHRASE_SESSION_KEY = "jp_rephrase_pending"


def _restore_journey_rephrase_from_session() -> None:
    mem = journey_chatbot.state.setdefault("global", {})
    backup = session.get(JP_REPHRASE_SESSION_KEY)
    if not isinstance(backup, dict):
        return
    if mem.get("pending_location_rephrase_for") in ("from", "to"):
        return
    side = backup.get("pending_location_rephrase_for")
    if side not in ("from", "to"):
        return
    mem["pending_location_rephrase_for"] = side
    mem["journey_planning_active"] = True
    for k in ("fromQuery", "toQuery", "fromLocationId", "toLocationId"):
        if k in backup:
            mem[k] = backup[k]


def _sync_journey_rephrase_session(jp_state: dict) -> None:
    if jp_state.get("pending_location_rephrase_for") in ("from", "to"):
        session[JP_REPHRASE_SESSION_KEY] = {
            "pending_location_rephrase_for": jp_state["pending_location_rephrase_for"],
            "fromQuery": jp_state.get("fromQuery"),
            "toQuery": jp_state.get("toQuery"),
            "fromLocationId": jp_state.get("fromLocationId"),
            "toLocationId": jp_state.get("toLocationId"),
        }
    elif JP_REPHRASE_SESSION_KEY in session:
        session.pop(JP_REPHRASE_SESSION_KEY, None)


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


def _load_stop_prefs(username: str) -> tuple:
    """Load stop preferences from DB.

    Returns (last_chosen_stop_id, frequent_stops_dict).
    Called once per chat request so the in-memory state is warm on first use
    after a server restart.
    """
    with get_db() as conn:
        row = conn.execute(
            'SELECT last_chosen_stop_id, frequent_stops FROM stop_preferences WHERE username = ?',
            (username,)
        ).fetchone()
    if row is None:
        return (None, {})
    return (row['last_chosen_stop_id'], json.loads(row['frequent_stops'] or '{}'))


def _save_stop_prefs(username: str, last_chosen_stop_id, frequent_stops: dict) -> None:
    """Persist stop preferences to DB (upsert)."""
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO stop_preferences (username, last_chosen_stop_id, frequent_stops)
               VALUES (?, ?, ?)
               ON CONFLICT(username) DO UPDATE SET
                 last_chosen_stop_id = excluded.last_chosen_stop_id,
                 frequent_stops      = excluded.frequent_stops''',
            (username, last_chosen_stop_id, json.dumps(frequent_stops))
        )
        conn.commit()


def _load_journey_history(username: str) -> dict:
    """Load journey location history from DB for the given user."""
    with get_db() as conn:
        row = conn.execute(
            'SELECT journey_history FROM stop_preferences WHERE username = ?',
            (username,)
        ).fetchone()
    if row is None:
        return {}
    return json.loads(row['journey_history'] or '{}')


def _save_journey_history(username: str, history: dict) -> None:
    """Persist journey history to DB (upsert, preserving other preference columns)."""
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO stop_preferences (username, journey_history)
               VALUES (?, ?)
               ON CONFLICT(username) DO UPDATE SET
                 journey_history = excluded.journey_history''',
            (username, json.dumps(history))
        )
        conn.commit()


def _get_shortcuts_for_user(username: str) -> dict:
    """Fetch all shortcuts for a user from the database as a {key: value} dict."""
    with get_db() as conn:
        rows = conn.execute(
            'SELECT key, value FROM shortcuts WHERE username = ?', (username,)
        ).fetchall()
    return {row['key']: row['value'] for row in rows}


def _apply_shortcuts_to_text(username: str, text: str) -> str:
    """
    Apply user-defined shortcuts to a free-text string.
    Replaces whole-word matches of each shortcut (case-insensitive).
    """
    if not text:
        return text

    shortcuts_for_user = _get_shortcuts_for_user(username)
    if not shortcuts_for_user:
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
        with get_db() as conn:
            rows = conn.execute(
                'SELECT key, value FROM shortcuts WHERE username = ? ORDER BY key',
                (username,)
            ).fetchall()
        return jsonify([{'key': r['key'], 'value': r['value']} for r in rows])

    data = request.get_json(force=True) or {}
    key = (data.get('key') or '').strip()
    value = (data.get('value') or '').strip()

    if not key or not value:
        return jsonify({'error': 'Both key and value are required.'}), 400

    with get_db() as conn:
        conn.execute(
            '''INSERT INTO shortcuts (username, key, value) VALUES (?, ?, ?)
               ON CONFLICT(username, key) DO UPDATE SET value = excluded.value''',
            (username, key.lower(), value)
        )
        conn.commit()

    return jsonify({'ok': True, 'key': key.lower(), 'value': value})


@app.route('/shortcuts/<key>', methods=['DELETE'])
def delete_shortcut(key):
    """Delete a single shortcut for the current user."""
    username = _current_user_key()
    key_lower = (key or '').lower()

    with get_db() as conn:
        cursor = conn.execute(
            'DELETE FROM shortcuts WHERE username = ? AND key = ?',
            (username, key_lower)
        )
        conn.commit()

    if cursor.rowcount == 0:
        return jsonify({'error': 'Shortcut not found.'}), 404

    return jsonify({'ok': True})


@app.route('/login', methods=['POST'])
def login():
    """Login or create account. New accounts: username > 3 chars, password >= 8 chars."""
    data = request.get_json(force=True) or {}
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password are required.'}), 400

    with get_db() as conn:
        row = conn.execute(
            'SELECT password_hash FROM users WHERE username = ?', (username,)
        ).fetchone()

    is_new_user = row is None
    if is_new_user:
        if len(username) <= 3:
            return jsonify({'error': 'Username must be longer than 3 characters.'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters.'}), 400
        password_hash = generate_password_hash(password)
        with get_db() as conn:
            conn.execute(
                'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
                (username, password_hash, datetime.utcnow().isoformat())
            )
            conn.commit()
    else:
        if not check_password_hash(row['password_hash'], password):
            return jsonify({'error': 'Incorrect password.'}), 400

    session['username'] = username
    return jsonify({'username': username})


@app.route('/logout', methods=['POST'])
def logout():
    """Log out the current user."""
    session.pop('username', None)
    return jsonify({'ok': True})


@app.route('/new_chat', methods=['POST'])
def new_chat():
    """
    Reset all conversational state for the current user.

    Called by the frontend "New chat" button so that intents, slots, and all
    journey-planning context are cleared and the next message starts a fresh
    conversation.
    """
    user_key = _current_user_key()

    # Reset traffic assistant FSM + per-user conversation state.
    traffic_chatbot.reset_user(user_key)

    # Reset journey planner's internal state.
    journey_chatbot.reset()
    session.pop(JP_REPHRASE_SESSION_KEY, None)

    return jsonify({'ok': True})


def _journey_planner_entities(state: dict) -> dict:
    """Build entities dict for the info panel from journey planner state."""
    entities = {}
    if not state:
        return entities
    if state.get("fromQuery"):
        entities["origin"] = state["fromQuery"]
    if state.get("toQuery"):
        entities["destination"] = state["toQuery"]
    when = state.get("when")
    if when and isinstance(when, dict):
        dt = when.get("datetime")
        if hasattr(dt, "strftime"):
            entities["time"] = dt.strftime("%H:%M")
            entities["date"] = dt.strftime("%Y-%m-%d")
        elif dt is not None:
            entities["time"] = str(dt)
        time_is = when.get("timeIs")
        if time_is:
            entities["time_preference"] = time_is.lower()
    if state.get("fromLocationId"):
        entities["from_id"] = state["fromLocationId"]
    if state.get("toLocationId"):
        entities["to_id"] = state["toLocationId"]
    # Include LLM-extracted TfL API parameters
    if state.get("_nlp_via"):
        entities["via"] = state["_nlp_via"]
    if state.get("_nlp_mode"):
        entities["mode"] = state["_nlp_mode"]
    if state.get("_nlp_journey_preference"):
        entities["journey_preference"] = state["_nlp_journey_preference"]
    return entities


def _build_journey_jsonify(jp_response: dict, entities: dict, confidence: float = 0.95) -> dict:
    """Build the standard JSON response for journey planner replies."""
    resp = {
        "response": jp_response.get("reply", ""),
        "journeys": jp_response.get("journeys", []),
        "tfl_journey_url": jp_response.get("tfl_journey_url"),
        "disambiguation": jp_response.get("disambiguation", False),
        "journey_disambiguation": jp_response.get("place_disambiguation"),
        "confirm_pins": jp_response.get("confirm_pins"),
        "state": jp_response.get("state", {}),
        "intent": "journey_planner",
        "entities": entities,
        "confidence": confidence,
    }
    # Journey history is now persisted server-side in SQLite.
    # No need to send it back to the frontend.
    jp_state = jp_response.get("state", {})
    _sync_journey_rephrase_session(jp_state)
    return resp


def _looks_like_journey_message(text: str) -> bool:
    """
    Heuristic: decide if a free-text message is asking to plan a journey.
    The word "plan" (e.g. "plan a journey", "plan my journey") should trigger
    Journey Planner intent, but timetable-style phrases like "bus times from X
    to Y" should stay with the timetable intent instead of being treated as
    journey planning.
    """
    t = (text or '').lower().strip()
    if not t:
        return False

    # Explicitly exclude timetable-style queries from journey-planner routing.
    timetable_phrases = [
        'bus times',
        'train times',
        'tube times',
        'timetable',
        'next bus',
        'next train',
        'bus or train times',
        'train or bus times',
    ]
    if any(p in t for p in timetable_phrases):
        return False

    # Strong journey-planner cues.
    if 'plan a journey' in t or 'plan journey' in t:
        return True
    if 'plan' in t and (
        'journey' in t or 'trip' in t or 'route' in t or 'get to' in t
    ):
        return True

    # Any message with both "from" and "to" that isn't a timetable query
    # is almost certainly a journey request ("from X to Y" in any word order).
    if ' from ' in t and ' to ' in t:
        return True

    # Destination-only journey phrases.
    dest_phrases = [
        'take me to', 'directions to', 'navigate to',
        'get me to', 'need to get to', 'route to',
        'go to', 'get to',
    ]
    if any(p in t for p in dest_phrases):
        return True

    # "take me from X", "directions from X to Y", "route from X to Y" etc.
    action_phrases = [
        'take me from', 'directions from', 'navigate from',
        'route from', 'need to get from', 'how do i get from',
        'how to get from', 'get from',
    ]
    if any(p in t for p in action_phrases):
        return True

    return False


def _maybe_save_journey_history(username: str, jp_response: dict) -> None:
    """If the journey planner updated history this request, persist it to SQLite."""
    if not username:
        return
    jp_state = jp_response.get("state", {})
    if jp_state.get("_updated_journey_history"):
        _save_journey_history(username, jp_state.get("_user_journey_history", {}))


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

        # Browser geolocation (sent from frontend)
        user_lat = data.get('userLat')
        user_lon = data.get('userLon')

        # Pin-confirmation submission: user dragged/accepted the origin+destination pins.
        # Update stored location IDs with the (possibly adjusted) coordinates and mark
        # pins as confirmed so the next pass through _continue_planning skips the map step.
        confirm_pins = data.get('confirmPins')  # {from: {lat,lon}, to: {lat,lon}}
        if confirm_pins and isinstance(confirm_pins, dict):
            jp_state_pre = journey_chatbot.state.setdefault("global", {})
            cf = confirm_pins.get('from') or {}
            ct = confirm_pins.get('to') or {}
            try:
                if cf.get('lat') is not None and cf.get('lon') is not None:
                    jp_state_pre['fromLocationId'] = f"{float(cf['lat'])},{float(cf['lon'])}"
                if ct.get('lat') is not None and ct.get('lon') is not None:
                    jp_state_pre['toLocationId'] = f"{float(ct['lat'])},{float(ct['lon'])}"
            except (TypeError, ValueError):
                pass
            jp_state_pre['_pins_confirmed'] = True

        # Apply user-defined shortcuts before routing the message.
        user_key = _current_user_key()
        user_message = _apply_shortcuts_to_text(user_key, user_message_raw)
        from_text = _apply_shortcuts_to_text(user_key, from_text_raw)
        to_text = _apply_shortcuts_to_text(user_key, to_text_raw)

        now = datetime.utcnow()

        username = session.get('username')

        # Restore stop preferences from DB into the in-memory state tracker.
        # inject_preferences() is a no-op if the user already has a live session,
        # so this only matters after a server restart.
        if username:
            last_stop_id, freq_stops = _load_stop_prefs(username)
            traffic_chatbot.inject_user_preferences(
                user_key, last_stop_id, freq_stops
            )

        # If we asked the user to rephrase a location, restore that flow from
        # session when in-memory state was lost (e.g. server reload).
        _restore_journey_rephrase_from_session()

        # Inject geolocation and journey history into journey planner state so
        # the disambiguation engine can use them for scoring.
        # Journey history is now loaded from SQLite rather than localStorage.
        jp_state = journey_chatbot.state.setdefault("global", {})
        if user_lat is not None and user_lon is not None:
            try:
                jp_state["_user_lat"] = float(user_lat)
                jp_state["_user_lon"] = float(user_lon)
            except (TypeError, ValueError):
                pass
        if username:
            jp_state["_user_journey_history"] = _load_journey_history(username)

        # Run the NLP pipeline once per request so we can:
        # - detect journey origin/destination slots even when the text does not
        #   obviously look like a journey query, and
        # - reuse these slots for either the journey planner or the traffic bot.
        nlp_result = None
        nlp_origin = None
        nlp_destination = None
        nlp_entities = {}
        try:
            nlp_result = traffic_chatbot.nlp.process(user_message)
            nlp_entities = nlp_result.get("entities", {}) if isinstance(nlp_result, dict) else {}
            nlp_origin = nlp_entities.get("origin")
            nlp_destination = nlp_entities.get("destination")
            if nlp_origin or nlp_destination:
                print(f"[app] NLP slots: origin={nlp_origin!r} destination={nlp_destination!r}")
        except Exception as _nlp_err:
            print(f"[app] NLP extraction error (non-fatal): {_nlp_err}")

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
            _sync_journey_rephrase_session(jp_response.get("state", {}))
            _maybe_save_journey_history(username, jp_response)
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
        if (
            jp_state.get("journey_planning_active")
            or jp_state.get("fromOptions")
            or jp_state.get("toOptions")
            or jp_state.get("pending_location_rephrase_for") in ("from", "to")
            or jp_state.get("_pins_confirmed")  # pin confirmation just submitted
        ):
            jp_response = journey_chatbot.handle_message(
                user_message,
                now=now,
                username=username,
                nlp_origin=nlp_origin,
                nlp_destination=nlp_destination,
                nlp_entities=nlp_entities,
            )
            # Keep the info panel consistent with the NLP output by merging
            # journey-planner state entities with LLM context fields.
            jp_entities = _journey_planner_entities(jp_response.get("state", {}))
            for k, v in nlp_entities.items():
                if k.startswith("llm_") or k in ("route_preference", "accessibility", "time_preference", "origin", "destination", "via", "mode", "journey_preference", "date", "time"):
                    jp_entities[k] = v
            _maybe_save_journey_history(username, jp_response)
            return jsonify(_build_journey_jsonify(jp_response, jp_entities))

        # 3) If the free-text clearly looks like a journey query OR the NLP
        #    extractor found a journey origin/destination (even if only one of
        #    them), route to the journey planner. The JourneyChatbot will reuse
        #    any provided slot and ask targeted follow-up questions for the
        #    missing side (e.g. "Where are you travelling from?" or "Where are
        #    you travelling to?").
        looks_like_journey = _looks_like_journey_message(user_message)
        nlp_intent = (nlp_result or {}).get("intent") if isinstance(nlp_result, dict) else None
        has_partial_journey_slots = bool(
            (nlp_origin and not nlp_destination) or (nlp_destination and not nlp_origin)
        )
        should_use_journey_planner = bool(
            looks_like_journey
            or nlp_intent == "journey_planning"
            or has_partial_journey_slots
        )

        if should_use_journey_planner:
            jp_response = journey_chatbot.handle_message(
                user_message,
                now=now,
                username=username,
                nlp_origin=nlp_origin,
                nlp_destination=nlp_destination,
                nlp_entities=nlp_entities,
            )
            # Merge journey planner state entities with LLM-extracted entities
            jp_entities = _journey_planner_entities(jp_response.get("state", {}))
            # Include LLM context fields (urgency, mood, utterance_type, preferences)
            for k, v in nlp_entities.items():
                if k.startswith("llm_") or k in ("route_preference", "accessibility", "time_preference"):
                    jp_entities[k] = v
            _maybe_save_journey_history(username, jp_response)
            return jsonify(_build_journey_jsonify(jp_response, jp_entities))

        # 4) Fallback: use the existing traffic chatbot for everything else.
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        traffic_response = traffic_chatbot.process_message(
            user_message,
            user_key=_current_user_key(),
            username=username,
            user_lat=user_lat,
            user_lon=user_lon,
        )

        # Persist any updated stop preferences back to SQLite so they survive
        # a server restart.  Only save for logged-in users.
        if username:
            # Read from _user_preferences — the authoritative store updated on
            # every resolve path (auto-resolve, towards-match, and manual pick).
            # state_tracker is only updated on manual picks, so using it here
            # would silently drop auto-resolved preferences.
            last_stop_id, freq_stops = traffic_chatbot.get_user_preferences(user_key)
            _save_stop_prefs(username, last_stop_id, freq_stops)

        # Merge LLM context fields into the traffic response entities
        traffic_entities = traffic_response.get('entities', {})
        for k, v in nlp_entities.items():
            if k.startswith("llm_") or k in ("route_preference", "accessibility", "time_preference"):
                traffic_entities[k] = v

        return jsonify({
            'response': traffic_response['message'],
            'intent': traffic_response.get('intent', 'unknown'),
            'entities': traffic_entities,
            'confidence': traffic_response.get('confidence', 0.0),
            'journeys': [],
            'disambiguation': False,
            'timetable': traffic_response.get('timetable'),
            'disruption': traffic_response.get('disruption'),
            'timetable_disambiguation': traffic_response.get('timetable_disambiguation'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Photo input (LandmarkLens integration)
# ---------------------------------------------------------------------------

@app.route('/landmark_status', methods=['GET'])
def landmark_status():
    """Whether the image-recognition service is reachable, for the UI to gate on."""
    try:
        return jsonify({'available': True, 'url': landmark_client.api_url(),
                        'service': landmark_client.health()})
    except landmark_client.LandmarkLensError as exc:
        return jsonify({'available': False, 'url': landmark_client.api_url(),
                        'error': str(exc)})


@app.route('/chat_photo', methods=['POST'])
def chat_photo():
    """Journey planning from a photo instead of (or alongside) typed text.

    The photo is sent to LandmarkLens, which returns a recognised
    `location_name`. That name is then injected into exactly the same slot the
    typed-text NLP pipeline fills -- `nlp_origin` or `nlp_destination` -- and
    handed to the identical `journey_chatbot.handle_message` call that /chat
    uses, so the journey flows through disambiguation, TfL lookup and rendering
    unchanged.

    Multipart form fields:
        image     (file, required)  the photo
        role      "to" (default) | "from"   which side of the journey it is
        message   optional accompanying text, e.g. "from Camden Town"
        date, time, userLat, userLon   as per /chat
    """
    try:
        file_storage = request.files.get('image') or request.files.get('photo')
        if file_storage is None or not file_storage.filename:
            return jsonify({'error': "no image uploaded (expected form field 'image')"}), 400

        image_bytes = file_storage.read()
        if not image_bytes:
            return jsonify({'error': 'uploaded image was empty'}), 400

        role = (request.form.get('role') or 'to').strip().lower()
        if role not in ('to', 'from'):
            role = 'to'

        # 1) Recognise the location from the photo.
        try:
            result = landmark_client.recognise(
                image_bytes, filename=file_storage.filename)
        except landmark_client.LandmarkLensError as exc:
            return jsonify({
                'response': ("I couldn't analyse that photo right now - the "
                             "image recognition service isn't responding. You "
                             "can still type the location instead."),
                'error': str(exc),
                'intent': 'photo_location',
                'entities': {},
                'confidence': 0.0,
                'journeys': [],
                'disambiguation': False,
            }), 502

        photo_meta = {
            'label': result.label,
            'class': result.class_slug,
            'location_name': result.location_name,
            'confidence': round(result.confidence, 4),
            'resolvable': result.resolvable,
            'low_confidence': result.low_confidence,
            'heatmap_url': result.heatmap_url,
            'role': role,
        }

        # 2) A recognised *kind* of place (a bus stop, a roundel) or a weak
        #    prediction is not a location we can route to. Say so and let the
        #    user type the specific name, rather than planning to a guess.
        if not result.usable:
            if result.low_confidence:
                reply = (f"I'm not confident about that photo - my best guess "
                         f"is {result.describe()}. Could you type the location "
                         f"name instead?")
            else:
                reply = (f"That looks like {result.label.lower()}, but that "
                         f"tells me the kind of place, not which one. "
                         f"Which {result.location_name.lower()} is it?")
            return jsonify({
                'response': reply,
                'intent': 'photo_location',
                'entities': {'photo_label': result.label},
                'confidence': result.confidence,
                'journeys': [],
                'disambiguation': False,
                'photo': photo_meta,
            })

        # 3) Run the normal NLP pipeline over any accompanying text, then let
        #    the photo win for its side of the journey. This is the whole
        #    integration: `location_name` becomes an ordinary location entity.
        user_key = _current_user_key()
        username = session.get('username')
        message = _apply_shortcuts_to_text(
            user_key, (request.form.get('message') or '').strip())
        now = datetime.utcnow()

        nlp_entities, nlp_origin, nlp_destination = {}, None, None
        if message:
            try:
                nlp_result = traffic_chatbot.nlp.process(message)
                nlp_entities = nlp_result.get('entities', {}) if isinstance(nlp_result, dict) else {}
                nlp_origin = nlp_entities.get('origin')
                nlp_destination = nlp_entities.get('destination')
            except Exception as nlp_err:
                print(f"[app] photo: NLP extraction error (non-fatal): {nlp_err}")

        if role == 'from':
            nlp_origin = result.location_name
        else:
            nlp_destination = result.location_name
        nlp_entities = dict(nlp_entities)
        nlp_entities['origin'] = nlp_origin
        nlp_entities['destination'] = nlp_destination
        nlp_entities['source'] = 'photo'
        print(f"[app] photo slots: origin={nlp_origin!r} "
              f"destination={nlp_destination!r} "
              f"(from {result.class_slug} @ {result.confidence:.2f})")

        # Same geolocation / history injection the text path performs.
        jp_state = journey_chatbot.state.setdefault('global', {})
        try:
            if request.form.get('userLat') and request.form.get('userLon'):
                jp_state['_user_lat'] = float(request.form['userLat'])
                jp_state['_user_lon'] = float(request.form['userLon'])
        except (TypeError, ValueError):
            pass
        if username:
            jp_state['_user_journey_history'] = _load_journey_history(username)

        # 4) Identical downstream call to the typed-text path.
        jp_response = journey_chatbot.handle_message(
            message,
            now=now,
            username=username,
            nlp_origin=nlp_origin,
            nlp_destination=nlp_destination,
            nlp_entities=nlp_entities,
        )

        jp_entities = _journey_planner_entities(jp_response.get('state', {}))
        for k, v in nlp_entities.items():
            if k.startswith('llm_') or k in ('route_preference', 'accessibility',
                                             'time_preference'):
                jp_entities[k] = v
        jp_entities['photo_location'] = result.location_name
        _maybe_save_journey_history(username, jp_response)

        payload = _build_journey_jsonify(jp_response, jp_entities,
                                         confidence=result.confidence)
        payload['photo'] = photo_meta
        return jsonify(payload)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    """Return the saved conversation list for the current user."""
    username = session.get('username')
    if not username:
        return jsonify([])
    with get_db() as conn:
        row = conn.execute(
            'SELECT data FROM chat_history WHERE username = ?', (username,)
        ).fetchone()
    if row is None:
        return jsonify([])
    return jsonify(json.loads(row['data'] or '[]'))


@app.route('/chat_history', methods=['PUT'])
def save_chat_history():
    """Persist the full conversation list for the current user."""
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a JSON array'}), 400
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO chat_history (username, data) VALUES (?, ?)
               ON CONFLICT(username) DO UPDATE SET data = excluded.data''',
            (username, json.dumps(data))
        )
        conn.commit()
    return jsonify({'ok': True})


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
