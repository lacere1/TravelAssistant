from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import os
import re

from chatbot import TrafficChatbot
from journey_planner import JourneyChatbot

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

USERS = {}

traffic_chatbot = TrafficChatbot()
journey_chatbot = JourneyChatbot()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')


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


@app.route('/me', methods=['GET'])
def me():
    username = session.get('username')
    return jsonify({'username': username})


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'ok': True})


def _looks_like_journey_message(text: str) -> bool:
    t = (text or '').lower()
    if ' from ' in t and ' to ' in t:
        return True
    if 'plan a journey' in t or 'plan journey' in t:
        return True
    return False


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_message = (data.get('message') or '').strip()
        from_text = (data.get('from') or '').strip()
        to_text = (data.get('to') or '').strip()

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        username = session.get('username')

        if from_text and to_text:
            jp_response = journey_chatbot.handle_structured_journey(
                from_text=from_text,
                to_text=to_text,
                now=datetime.utcnow(),
                username=username,
            )
            return jsonify({
                'response': jp_response.get('reply', ''),
                'journeys': jp_response.get('journeys', []),
                'intent': 'journey_planner',
                'entities': {'from': from_text, 'to': to_text},
                'confidence': 1.0,
            })

        if _looks_like_journey_message(user_message):
            jp_response = journey_chatbot.handle_message(user_message, now=datetime.utcnow(), username=username)
            return jsonify({
                'response': jp_response.get('reply', ''),
                'journeys': jp_response.get('journeys', []),
                'intent': 'journey_planner',
                'entities': {},
                'confidence': 0.95,
            })

        traffic_response = traffic_chatbot.process_message(user_message, username=username)
        return jsonify({
            'response': traffic_response['message'],
            'intent': traffic_response.get('intent', 'unknown'),
            'entities': traffic_response.get('entities', {}),
            'confidence': traffic_response.get('confidence', 0.0),
            'timetable': traffic_response.get('timetable'),
            'disruption': traffic_response.get('disruption'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Travel assistant'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
