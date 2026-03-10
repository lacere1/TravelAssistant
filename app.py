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
    """Main page with chat + journey planner interface"""
    return render_template('index.html')


@app.route('/login', methods=['GET'])
def login_page():
    """Dedicated login / sign-up page."""
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    """Login or create account."""
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


@app.route('/me', methods=['GET'])
def me():
    """Return the currently logged-in user (if any)."""
    username = session.get('username')
    return jsonify({'username': username})


@app.route('/chat', methods=['POST'])
def chat():
    """Unified chat endpoint for both the traffic assistant and journey planner."""
    try:
        data = request.get_json(force=True) or {}
        user_message = (data.get('message') or '').strip()

        from_text = (data.get('from') or '').strip()
        to_text = (data.get('to') or '').strip()
        date_str = (data.get('date') or '').strip() or None
        time_str = (data.get('time') or '').strip() or None

        now = datetime.utcnow()
        username = session.get('username')

        if from_text and to_text:
            jp_response = journey_chatbot.handle_structured_journey(
                from_text=from_text,
                to_text=to_text,
                date_str=date_str,
                time_str=time_str,
                now=now,
                username=username,
            )
            return jsonify({
                'response': jp_response.get('reply', ''),
                'journeys': jp_response.get('journeys', []),
                'intent': 'journey_planner',
                'entities': {
                    'from': from_text,
                    'to': to_text,
                },
                'confidence': 1.0,
            })

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        traffic_response = traffic_chatbot.process_message(user_message, username=username)

        return jsonify({
            'response': traffic_response['message'],
            'intent': traffic_response.get('intent', 'unknown'),
            'entities': traffic_response.get('entities', {}),
            'confidence': traffic_response.get('confidence', 0.0),
            'journeys': [],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Travel assistant'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
