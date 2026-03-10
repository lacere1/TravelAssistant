from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import os
import re

from chatbot import TrafficChatbot
from journey_planner import JourneyChatbot

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

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


@app.route('/chat', methods=['POST'])
def chat():
    """Unified chat endpoint for both the traffic assistant and journey planner."""
    try:
        data = request.get_json(force=True) or {}
        user_message = (data.get('message') or '').strip()

        from_text = (data.get('from') or '').strip()
        to_text = (data.get('to') or '').strip()

        now = datetime.utcnow()

        if from_text and to_text:
            jp_response = journey_chatbot.handle_message(
                f"Plan a journey from {from_text} to {to_text}",
                now=now
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

        traffic_response = traffic_chatbot.process_message(user_message)

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
