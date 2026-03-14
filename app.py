from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import os

from chatbot import TrafficChatbot
from journey_planner import JourneyChatbot

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

USERS = {}
traffic_chatbot = TrafficChatbot()
journey_chatbot = JourneyChatbot()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')


@app.route('/me', methods=['GET'])
def me():
    return jsonify({'username': session.get('username')})


@app.route('/login', methods=['POST'])
def login():
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
    session.pop('username', None)
    return jsonify({'ok': True})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_message = (data.get('message') or '').strip()
        from_text = (data.get('from') or '').strip()
        to_text = (data.get('to') or '').strip()
        from_id = (data.get('fromId') or '').strip() or None
        to_id = (data.get('toId') or '').strip() or None
        date_str = (data.get('date') or '').strip() or None
        time_str = (data.get('time') or '').strip() or None
        now = datetime.utcnow()
        username = session.get('username')

        # Structured journey inputs
        if from_text and to_text:
            jp = journey_chatbot.handle_structured_journey(
                from_text=from_text, to_text=to_text,
                from_id=from_id, to_id=to_id,
                date_str=date_str, time_str=time_str,
                now=now, username=username,
            )
            return jsonify({
                'response': jp.get('reply', ''),
                'journeys': jp.get('journeys', []),
                'intent': 'journey_planner',
                'entities': {'from': from_text, 'to': to_text},
                'confidence': 1.0,
                'disambiguation': jp.get('disambiguation', False),
            })

        # Free-text journey
        t = (user_message or '').lower()
        if ' from ' in t and ' to ' in t:
            jp = journey_chatbot.handle_message(user_message, now=now, username=username)
            return jsonify({
                'response': jp.get('reply', ''),
                'journeys': jp.get('journeys', []),
                'intent': 'journey_planner',
                'entities': {},
                'confidence': 0.95,
                'disambiguation': jp.get('disambiguation', False),
            })

        # Traffic chatbot
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        resp = traffic_chatbot.process_message(user_message, username=username)
        return jsonify({
            'response': resp['message'],
            'intent': resp.get('intent', 'unknown'),
            'entities': resp.get('entities', {}),
            'confidence': resp.get('confidence', 0.0),
            'journeys': [],
            'disambiguation': False,
            'timetable': resp.get('timetable'),
            'disruption': resp.get('disruption'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
