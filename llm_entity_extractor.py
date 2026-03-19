"""
LLM-based Entity Extractor using Anthropic Claude API.

Replaces the hybrid SpaCy + regex NER pipeline with a single LLM call
that extracts structured journey-planning entities from free-text user input.

Extracted entity categories:
  1. Core journey slots: origin, destination, date, time, time_preference
  2. Travel preferences: travel_mode, accessibility, route_preference
  3. Context/sentiment: urgency, mood, utterance_type

Falls back gracefully when the API key is missing or the call fails,
returning an empty entity dict so the rest of the pipeline still works.
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False
    print("[LLM-NER] anthropic package not installed. Run: pip install anthropic")


# ── Extraction prompt ────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise entity-extraction engine for a London travel assistant chatbot.

Given the user's message, extract ALL relevant entities into a flat JSON object.
Only include keys whose values you can confidently extract from the text.
Do NOT guess or hallucinate values that aren't present.

### Entity schema

**Core journey slots**
- "origin"            : string – where the user is travelling FROM (place name, postcode, or coordinates)
- "destination"       : string – where the user is travelling TO (place name, postcode, or coordinates)
- "via"               : string – an intermediate waypoint the user wants to travel THROUGH (e.g. "via King's Cross", "through Baker Street"). Can be a place name, UK postcode, or coordinates.
- "date"              : string – travel date in natural language (e.g. "tomorrow", "next Monday", "25th March")
- "time"              : string – travel time in natural language (e.g. "9am", "around 3pm", "morning")
- "time_preference"   : "departing" | "arriving" – whether the stated time is when the user wants to LEAVE or ARRIVE

**Travel mode preferences**
- "mode"              : comma-separated string of transport modes the user wants to use. Pick ONLY from these exact TfL API mode identifiers: "bus", "overground", "national-rail", "tube", "coach", "dlr", "cable-car", "tram", "river-bus", "walking", "cycle", "elizabeth-line". Examples: "tube" if user says "by tube"; "bus" if user says "by bus"; "national-rail" for trains. If user says "no buses" or "avoid buses", list all modes EXCEPT "bus". Do NOT include "walking" alongside other modes — TfL includes walking connections automatically.
- "accessibility"     : string – any accessibility needs mentioned (e.g. "step-free", "wheelchair", "no stairs", "no escalators")

**Journey preference**
- "journey_preference": "leastinterchange" | "leasttime" | "leastwalking" – how the user wants to optimise their journey. "leastinterchange" = fewest changes; "leasttime" = fastest route; "leastwalking" = least walking. Infer from phrases like "quickest" → "leasttime", "fewest changes" → "leastinterchange", "least walking" / "minimal walking" → "leastwalking".

**Transport-specific (for timetable/disruption queries)**
- "bus_route"         : string – specific bus route number (e.g. "83", "N7")
- "tube_line"         : string – specific tube/rail line name (e.g. "Northern", "Jubilee", "Elizabeth")
- "location"          : string – a general location mentioned (for timetable/disruption queries, not journey origin/dest)
- "near_area"         : string – a broader area, neighbourhood, or landmark the user mentions to indicate WHERE THE STOP IS LOCATED. Extract from spatial-proximity phrases like "in Kingsbury", "near Wembley", "by the station", "around Clapham". This is NOT the direction buses travel — it's the physical area context for disambiguating which specific stop the user means. For example: "bus times at Lavender Avenue in Kingsbury" → location="Lavender Avenue", near_area="Kingsbury". "the stop near Wembley Park" → near_area="Wembley Park". Only extract if the user explicitly mentions an area/neighbourhood/landmark to narrow down a stop's location.
- "towards"           : string – the direction or destination the user wants the VEHICLE TO TRAVEL TOWARDS (applies to buses, trains, tubes, overground, DLR, etc.). Extract ONLY from directional travel phrases like "towards Burnt Oak", "going towards Wembley", "heading to Stanmore", "in the direction of Edgware", "towards Watford", "towards Brighton". This is about WHERE THE VEHICLE IS GOING, not where the stop is. For example: "bus times at Lavender Avenue towards Burnt Oak" → location="Lavender Avenue", towards="Burnt Oak". "tube times at King's Cross towards Watford" → location="King's Cross", towards="Watford". "trains at Stratford towards Liverpool Street" → location="Stratford", towards="Liverpool Street". Do NOT use this for journey planner origin/destination — only for timetable queries where the user wants vehicles travelling in a specific direction.

**Context & sentiment**
- "urgency"           : "low" | "normal" | "high" – how urgent the request feels
- "mood"              : "neutral" | "frustrated" | "curious" | "polite" | "excited"
- "utterance_type"    : "question" | "command" | "statement" | "greeting" | "goodbye"

### Rules
1. Return ONLY valid JSON – no markdown, no explanation, no wrapping.
2. Omit keys that have no evidence in the user's text.
3. For "origin", "destination", and "via", extract the place name exactly as the user typed it.
4. If the user mentions a specific bus route number, put it in "bus_route" NOT "origin"/"destination".
5. If the user mentions a tube/train line name, put it in "tube_line".
6. "location" is for single-location queries like "bus times at Oxford Circus" – not for journey endpoints.
6b. "near_area" vs "towards" — these are DIFFERENT and must not be confused:
   near_area = WHERE THE STOP/STATION IS (physical location context, uses proximity words: in, near, by, around, at)
   towards   = WHERE THE VEHICLE IS GOING (travel direction, uses motion words: towards, going to, heading to, direction of)
   towards applies to ALL timetable modes: buses, trains, tubes, overground, DLR, etc.

   Examples of near_area (stop/station location):
   - "bus times at Lavender Avenue in Kingsbury"       → location="Lavender Avenue", near_area="Kingsbury"
   - "bus times at Lavender Avenue near Kingsbury"     → location="Lavender Avenue", near_area="Kingsbury"
   - "the stop near Wembley Park"                      → near_area="Wembley Park"
   - "Oxford Circus stop by Regent Street"             → location="Oxford Circus", near_area="Regent Street"

   Examples of towards (vehicle travel direction):
   - "bus times at Lavender Avenue towards Burnt Oak"      → location="Lavender Avenue", towards="Burnt Oak"
   - "buses going to Wembley from Lavender Avenue"         → location="Lavender Avenue", towards="Wembley"
   - "Lavender Avenue buses heading to Stanmore"           → location="Lavender Avenue", towards="Stanmore"
   - "tube times at King's Cross towards Watford"          → location="King's Cross", towards="Watford"
   - "trains at Stratford towards Liverpool Street"        → location="Stratford", towards="Liverpool Street"
   - "overground times at Highbury towards Clapham"        → location="Highbury", towards="Clapham"
   - "train timetable for Wembley Park to Euston"          → location="Wembley Park", towards="Euston"
   - "tube timetable for King's Cross to Morden"           → location="King's Cross", towards="Morden"
   - "I want the train timetable for Wembley Park to Euston" → location="Wembley Park", towards="Euston"
   NOTE: "timetable for X to Y" is NOT a journey planner query — X is the stop location, Y is the direction. Use location+towards, NOT origin+destination.

   Both can appear together:
   - "bus times at Lavender Avenue in Kingsbury towards Burnt Oak" → location="Lavender Avenue", near_area="Kingsbury", towards="Burnt Oak"
   - "tube at Finchley Road near Wembley towards Waterloo"         → location="Finchley Road", near_area="Wembley", towards="Waterloo"

   Do NOT extract near_area if the user only says a single location with no area context.
   Do NOT extract towards for journey planner queries (those use origin/destination instead).
   "timetable for X to Y" uses location+towards. "how do I get from X to Y" uses origin+destination.
7. A greeting like "hi" should return: {"utterance_type": "greeting", "mood": "polite"}
8. If the text has no extractable travel entities, return {}.
9. For "mode": only include if the user explicitly mentions transport modes. Use exact TfL API identifiers:
   - "by tube" → "tube"
   - "by bus" → "bus"
   - "by train" → "national-rail"
   - "by overground" → "overground"
   - "by DLR" → "dlr"
   - "by Elizabeth line" → "elizabeth-line"
   - "walk" or "on foot" → "walking"
   - "cycle" or "bike" → "cycle"
   - "by boat" / "river bus" → "river-bus"
   - "cable car" → "cable-car"
   - "no buses" → "tube,national-rail,overground,dlr,elizabeth-line,tram,river-bus,walking,cycle"
   - "public transport" → "bus,tube,national-rail,overground,dlr,elizabeth-line,tram"
   If the user doesn't specify a mode, omit this key entirely. Do NOT append "walking" to other modes — TfL adds walking connections automatically.
10. For "time_preference": "arrive by 6pm" → "arriving"; "leave at 9am" / "departing 9am" → "departing". Default to "departing" only if a time is given without arrival context.
11. For "via": only extract if the user explicitly says "via", "through", "stopping at", or "passing through" a location.
"""

_USER_PROMPT_TEMPLATE = 'Extract entities from this message:\n\n"{text}"'


class LLMEntityExtractor:
    """
    Uses Anthropic Claude to extract structured entities from user messages.
    Thread-safe with a simple in-memory cache to avoid duplicate API calls.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 512,
        temperature: float = 0.0,
        cache_ttl: int = 300,       # seconds
        cache_max_size: int = 200,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client: Optional[Any] = None
        self._available = False

        # Simple thread-safe LRU-ish cache: { text_lower: (timestamp, entities) }
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = cache_ttl
        self._cache_max = cache_max_size
        self._cache_lock = threading.Lock()

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Anthropic client."""
        if not _ANTHROPIC_AVAILABLE:
            print("[LLM-NER] Anthropic SDK not available – LLM extraction disabled.")
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            print("[LLM-NER] ANTHROPIC_API_KEY not set – LLM extraction disabled.")
            print("[LLM-NER] Add ANTHROPIC_API_KEY=sk-... to your .env file.")
            return

        try:
            self._client = anthropic.Anthropic(api_key=api_key)
            self._available = True
            print(f"[LLM-NER] Initialized with model={self._model}")
        except Exception as exc:
            print(f"[LLM-NER] Failed to initialize Anthropic client: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from user text via Claude API.

        Returns a dict of extracted entities (may be empty).
        Never raises – returns {} on any failure.
        """
        if not self._available or not text or not text.strip():
            return {}

        text_stripped = text.strip()
        cache_key = text_stripped.lower()

        # Check cache
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached:
                ts, entities = cached
                if time.time() - ts < self._cache_ttl:
                    return entities
                else:
                    del self._cache[cache_key]

        # Call the API
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": _USER_PROMPT_TEMPLATE.format(text=text_stripped),
                    }
                ],
            )

            raw_text = response.content[0].text.strip()

            # Parse JSON – handle models that sometimes wrap in ```json blocks
            json_text = raw_text
            if json_text.startswith("```"):
                # Strip markdown code fences
                lines = json_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                json_text = "\n".join(lines)

            entities = json.loads(json_text)

            if not isinstance(entities, dict):
                print(f"[LLM-NER] Expected dict, got {type(entities).__name__}")
                return {}

            # Store in cache
            with self._cache_lock:
                if len(self._cache) >= self._cache_max:
                    # Evict oldest entry
                    oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                    del self._cache[oldest_key]
                self._cache[cache_key] = (time.time(), entities)

            return entities

        except json.JSONDecodeError as exc:
            print(f"[LLM-NER] JSON parse error: {exc}")
            print(f"[LLM-NER] Raw response: {raw_text[:200]}")
            return {}
        except Exception as exc:
            print(f"[LLM-NER] API call failed: {exc}")
            return {}

    def clear_cache(self) -> None:
        """Clear the entity cache."""
        with self._cache_lock:
            self._cache.clear()


# ── Module-level singleton ───────────────────────────────────────────
_extractor_instance: Optional[LLMEntityExtractor] = None
_extractor_lock = threading.Lock()


def get_llm_extractor() -> LLMEntityExtractor:
    """Get or create the singleton LLMEntityExtractor."""
    global _extractor_instance
    if _extractor_instance is None:
        with _extractor_lock:
            if _extractor_instance is None:
                _extractor_instance = LLMEntityExtractor()
    return _extractor_instance
