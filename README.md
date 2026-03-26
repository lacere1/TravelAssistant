# London Travel Assistant

A conversational web app for London transport queries, powered by a multi-layer NLP pipeline and the TfL (Transport for London) API. Ask about bus times, plan a journey, check for disruptions, or get real-time arrival info — all through a natural language chat interface.

---

## Features

- **Journey Planner** — plan door-to-door journeys across London using the TfL Journey API, with support for departure/arrival time preferences, transport mode selection, and accessibility options.
- **Timetable & Arrivals** — query live bus and train arrival times at any stop.
- **Disruption Alerts** — get service disruption and status information.
- **Conversational NLP** — a fine-tuned intent classifier (sentence-transformers + logistic regression) routes messages to the right handler, with Claude (Anthropic API) used for entity extraction.
- **Disambiguation Engine** — when a place name matches multiple stops or locations, the app presents options and uses journey history + geolocation to rank them intelligently.
- **User Accounts** — register and log in to save preferences, shortcuts, and chat history across sessions (SQLite-backed).
- **Custom Shortcuts** — define personal shortcuts (e.g. `home` → `Neasden Station`) that are automatically expanded in your messages.
- **Google Places Autocomplete** — optional frontend autocomplete for location inputs.
- **Cross-platform** — run scripts provided for Windows (`run.bat`) and Unix/Mac (`run.sh`).

---

## Architecture

```
app.py                  Flask app, routing logic, SQLite persistence, REST API
chatbot.py              Traffic/timetable chatbot (FSM-based dialog manager)
journey_planner.py      TfL journey planning chatbot + TfL Journey API client
intent_classifier.py    Sentence-transformer + LogisticRegression intent classifier
nlp_processor.py        Main NLP pipeline (coordinates intent + entity extraction)
llm_entity_extractor.py Anthropic Claude-based entity extractor
ner_processor.py        SpaCy NER processor (fallback entity extraction)
disambiguation_engine.py Candidate ranking and location disambiguation
places_grounder.py      OSM / TfL place search and geocoding
journey_slot_extractor.py Slot filling for journey planning (origin, destination, time)
dialog_state.py         Per-user dialog state tracking
training_data.py        Labelled training examples for the intent classifier
transport_api.py        TfL REST API wrapper
tfl_stop_datasets.py    TfL stop data loader utilities
bus_stops.csv           Bus stop reference data
train_stops.csv         Train/tube stop reference data
tfl_bus_routes.txt      Bus route reference data
templates/              Jinja2 HTML templates (index.html, login.html)
static/                 Frontend assets (CSS, JS)
```

---

## Prerequisites

- Python 3.9+
- pip
- An **Anthropic API key** (for LLM entity extraction)
- A **TfL API key** (optional but recommended for higher rate limits — get one at [api.tfl.gov.uk](https://api.tfl.gov.uk))
- A **Google Places API key** (optional, enables location autocomplete in the UI)

---

## Setup

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd TravelAssistant
```

**2. Create and activate a virtual environment** (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

After installing, download the SpaCy language model used incase for backup NER:

```bash
python -m spacy download en_core_web_sm
```

**4. Configure environment variables**

Create a `.env` file in the project root:

```env
# Required — Anthropic Claude API key for entity extraction
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional — TfL API credentials (increases rate limits)
TFL_APP_ID=your_tfl_app_id
TFL_APP_KEY=your_tfl_app_key

# Optional — Google Places API key (enables location autocomplete)
GOOGLE_PLACES_API_KEY=your_google_places_key

# Optional — Change this in production
SECRET_KEY=your_flask_secret_key
```

**5. Run the app**

```bash
# Windows
run.bat

# Mac / Linux
bash run.sh

# Or directly
python app.py
```

The server starts on `http://localhost:5000`.

---

## Usage

Open `http://localhost:5000` in your browser. You can use the app as a guest or create an account to save preferences and history.

**Example queries:**

- `When is the next 43 bus from London Bridge?`
- `Plan a journey from Paddington to Canary Wharf arriving by 9am`
- `Is the Central line running?`
- `Bus times from home to work` *(with shortcuts configured)*
- `plan a journey to Kings Cross`

**Managing shortcuts:**

Go to Settings in the UI, or use the `/shortcuts` API directly to add key→value expansions for places you travel to often.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Main chat endpoint (accepts natural language or structured journey fields) |
| `GET/POST` | `/shortcuts` | List or create user shortcuts |
| `DELETE` | `/shortcuts/<key>` | Delete a shortcut |
| `POST` | `/login` | Login or register |
| `POST` | `/logout` | Logout |
| `GET` | `/me` | Get current logged-in user |
| `POST` | `/new_chat` | Reset conversation state |
| `GET/PUT` | `/chat_history` | Load or save conversation history |
| `GET` | `/suggest` | TfL place search autocomplete |
| `GET` | `/health` | Health check |

---

## Testing

Test sets and results are included in the project:

- `TravelAssistant_TestSet.xlsx` — input test cases
- `TravelAssistant_TestSet_COMPLETED.xlsx` — test cases with expected outputs
- `TravelAssistant_TestResults_Actual.xlsx` — actual outputs from evaluation runs
- `evaluate_classifier.py` — runs the intent classifier against the test set

To evaluate the classifier:

```bash
python evaluate_classifier.py
```

---

## Database

The app uses a local SQLite database (`travel_assistant.db`) created automatically on first run. It stores:

- `users` — account credentials (bcrypt-hashed passwords)
- `shortcuts` — per-user text shortcuts
- `stop_preferences` — last-used stop, frequently-used stops, and journey location history
- `chat_history` — saved conversation messages per user

---

## Dependencies

Key libraries used:

| Library | Purpose |
|---------|---------|
| Flask | Web framework |
| sentence-transformers | Intent classifier embeddings |
| scikit-learn | Logistic regression classifier |
| anthropic | LLM entity extraction (Claude) |
| spacy | NER fallback |
| requests | TfL API calls |
| python-dotenv | Environment variable loading |
| werkzeug | Password hashing |

See `requirements.txt` for pinned versions.

---

## License

This project is provided as-is for educational and personal use.
