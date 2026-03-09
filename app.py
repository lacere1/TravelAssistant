from flask import Flask, render_template, request, jsonify

from chatbot import TrafficChatbot

app = Flask(__name__)

chatbot = TrafficChatbot()


@app.route("/")
def index():
    """Single-page chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Very simple chat endpoint."""
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    reply = chatbot.reply(message)
    return jsonify({"response": reply})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)

