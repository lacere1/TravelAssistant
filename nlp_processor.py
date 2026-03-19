"""
NLP Processor — Unified interface for intent classification + entity extraction.

Architecture (upgraded):
  - Intent classification: sentence-transformers + LogisticRegression (IntentClassifier)
    with rule-based fallback for edge cases
  - NER: SpaCy (general entities) + domain-specific regex (NERProcessor)
  - CSV-backed stop name fuzzy matching for timetable refinement (preserved)
  - Backward-compatible API: process() returns the same dict shape as before

The old zero-shot transformer pipeline has been replaced. The new classifier
is 10-50x faster and more accurate on domain-specific intents.
"""
import re
import csv
import os
from difflib import SequenceMatcher
from typing import Dict, List, Any, Set, Optional, Tuple

# New components
from intent_classifier import IntentClassifier
from ner_processor import NERProcessor
from llm_entity_extractor import get_llm_extractor
from journey_slot_extractor import get_extractor as get_journey_slot_extractor
from tfl_stop_datasets import TRAIN_LINES, load_bus_routes, load_bus_stops, load_train_stations

try:
    # Optional NLTK + WordNet support for synonym expansion
    import nltk  # type: ignore[import]
    from nltk.corpus import wordnet as wn  # type: ignore[import]
    _NLTK_AVAILABLE = True
except Exception as e:
    print(f"NLTK not available for synonym expansion: {e}")
    wn = None  # type: ignore[assignment]
    _NLTK_AVAILABLE = False


class NLPProcessor:
    def __init__(self):
        """Initialize NLP models for intent classification and entity extraction"""
        print("Loading NLP models...")
        # Enable synonym expansion only if NLTK/WordNet are available
        self._synonym_expansion_enabled = _NLTK_AVAILABLE

        # ---- New: Fine-tuned intent classifier ----
        # Replaces the slow zero-shot classification pipeline
        self._intent_classifier = IntentClassifier()

        # ---- Backup: Hybrid SpaCy + regex NER ----
        # Create only if/when LLM entity extraction is unavailable.
        self._ner_fallback: Optional[NERProcessor] = None

        # ---- Backup: Rule-based journey slot extraction ----
        # Used when the LLM entity extractor is unavailable OR fails at runtime.
        self._journey_slot_extractor = get_journey_slot_extractor()

        # ---- New: LLM-based entity extraction (Anthropic Claude) ----
        self._llm_extractor = get_llm_extractor()
        if self._llm_extractor.available:
            print("[NLP] LLM entity extractor enabled (Anthropic Claude)")
        else:
            print("[NLP] LLM entity extractor unavailable – falling back to SpaCy + regex NER")

        # Legacy: keep the old zero-shot classifier as an optional deep fallback
        # (only loaded if the new classifier fails to initialize)
        self.intent_classifier = None  # old zero-shot pipeline (disabled by default)
        if not self._intent_classifier.is_ready:
            print("[NLP] New classifier not ready; attempting legacy zero-shot fallback...")
            try:
                from transformers import pipeline as hf_pipeline
                model_options = [
                    "typeform/distilbert-base-uncased-mnli",
                    "facebook/bart-large-mnli",
                ]
                for model_name in model_options:
                    try:
                        self.intent_classifier = hf_pipeline(
                            "zero-shot-classification",
                            model=model_name,
                            device=-1,
                        )
                        print(f"[NLP] Loaded legacy zero-shot model: {model_name}")
                        break
                    except Exception:
                        continue
            except ImportError:
                pass

        # Intent labels (used only by legacy zero-shot fallback if needed)
        # Restricted to the supported intents for this assistant.
        self.intent_labels = [
            "greeting",
            "goodbye",
            "ask_timetable",
            "ask_transit_disruption",
            "journey_planning",
        ]

        print("NLP models loaded successfully")

        # Pre-load stop name datasets for CSV-backed intent refinement.
        # This is intentionally not tied to NERProcessor so that NER is a
        # true fallback component.
        self._bus_stops: List[str] = load_bus_stops()
        self._train_stations: List[str] = load_train_stations()
        self._train_lines: List[str] = list(TRAIN_LINES)
        self._bus_routes: Set[str] = load_bus_routes()
        self._stops_loaded: bool = True
    
    def _expand_with_synonyms(self, tokens: List[str]) -> Set[str]:
        """
        Expand a list of tokens with WordNet synonyms.
        
        Returns a set containing the original tokens plus any synonym words
        (multi-word synonyms are split into individual tokens).
        If NLTK/WordNet are unavailable, this simply returns the original tokens.
        """
        expanded: Set[str] = set(t.lower() for t in tokens if t)
        
        if not self._synonym_expansion_enabled or not wn:
            return expanded
        
        for token in list(expanded):
            # Skip very short tokens to avoid noisy expansions
            if len(token) < 3:
                continue
            before_token = set(expanded)
            try:
                for synset in wn.synsets(token):  # type: ignore[union-attr]
                    for lemma in synset.lemmas():
                        name = lemma.name().replace("_", " ").lower()
                        if not name:
                            continue
                        # Split multi-word synonyms like "traffic jam"
                        for part in name.split():
                            if part:
                                expanded.add(part)
                added_for_token = sorted(expanded - before_token)
                if added_for_token:
                    print(f"[NLP] Synonyms added for '{token}': {added_for_token}")
            except LookupError:
                # WordNet data not downloaded/available; disable further attempts
                print("WordNet corpus not available; disabling synonym expansion.")
                self._synonym_expansion_enabled = False
                break
            except Exception:
                # Any other NLTK-related issue: fail gracefully and continue
                continue
        
        return expanded
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process user input to extract intent and entities.

        Uses the new fine-tuned IntentClassifier and hybrid SpaCy+regex NER,
        with rule-based overrides for edge cases and CSV-backed refinement.

        Args:
            text: User's natural language input

        Returns:
            dict: Contains intent, entities, and confidence score
        """
        original_text = text
        # Normalize text
        text = text.strip().lower()
        print(f"[NLP] Incoming text: {original_text!r}")

        # ---- Intent classification (new: fine-tuned classifier) ----
        intent, confidence = self._classify_intent(text)
        print(f"[NLP] Classifier intent: {intent}, confidence: {confidence:.3f}")

        # ---- Entity extraction ----
        # Primary: LLM-based extraction (Anthropic Claude)
        # Backup: JourneySlotExtractor + hybrid SpaCy+regex NER (runtime-fallback)
        entities: Dict[str, Any] = {}
        llm_ok = False
        if self._llm_extractor.available:
            try:
                llm_entities = self._llm_extractor.extract_entities(original_text)
                if isinstance(llm_entities, dict) and llm_entities:
                    print(f"[NLP] LLM extracted entities: {llm_entities}")
                    entities = self._normalize_llm_entities(llm_entities)
                    llm_ok = True
                else:
                    print("[NLP][LLM] Returned no entities → falling back to JourneySlotExtractor + NERProcessor.")
            except Exception as e:
                # Runtime failure (timeout/network/API/etc.) → fall back gracefully.
                print(f"[NLP][LLM] FAILED ({type(e).__name__}: {e}) → falling back to JourneySlotExtractor + NERProcessor.")

        if not llm_ok:
            # Journey slots first (origin/destination + optional Places grounding)
            try:
                journey_slots = self._journey_slot_extractor.extract_journey_slots(original_text)
                if journey_slots:
                    entities.update(journey_slots)
                    print(f"[NLP] JourneySlotExtractor entities: {journey_slots}")
            except Exception as e:
                print(f"[NLP][Fallback] JourneySlotExtractor failed ({type(e).__name__}: {e})")

            # General/domain entities via SpaCy + regex
            if self._ner_fallback is None:
                self._ner_fallback = NERProcessor()
            try:
                ner_entities = self._ner_fallback.extract_entities(original_text)
                if ner_entities:
                    # Don't overwrite journey slots already extracted above
                    for k, v in ner_entities.items():
                        if k not in entities:
                            entities[k] = v
            except Exception as e:
                print(f"[NLP][Fallback] NERProcessor failed ({type(e).__name__}: {e})")

            # Also run legacy regex extraction and merge (for non-journey patterns)
            legacy_entities = self._extract_entities(text)
            for key, value in legacy_entities.items():
                if key not in entities:
                    entities[key] = value
        # ---- CSV-backed refinement (preserved from original) ----
        intent, entities, confidence = self._refine_with_stop_datasets(
            original_text, intent, entities, confidence
        )

        # ---- Scope journey slots to journey-planning intent ----
        # Origin/destination can be noisy for non-journey queries. To keep them
        # from polluting other intents, only expose them when we're in an
        # explicit journey_planning flow.
        #
        # HOWEVER: if the LLM extracted origin or destination, trust it and
        # override the intent to journey_planning. The LLM is specifically
        # designed for journey entity extraction and is more reliable than
        # the sentence-transformer intent classifier for this purpose.
        if intent != "journey_planning":
            # Never let journey slots force us into journey planning for these intents.
            # For timetable/disruption queries we prefer keeping the current intent,
            # and we also strip journey slots to avoid polluting downstream handlers.
            protected_intents = {"ask_timetable", "ask_transit_disruption"}

            llm_has_journey_slots = entities.get("origin") or entities.get("destination")
            if llm_has_journey_slots and intent not in protected_intents:
                print(f"[NLP] LLM found journey slots but intent was '{intent}' — overriding to 'journey_planning'")
                intent = "journey_planning"
            else:
                # For timetable queries, "destination" often means "towards"
                # (e.g. "train timetable for Wembley Park to Euston" →
                # destination="Euston" should become towards="Euston").
                # Preserve it before stripping journey slots.
                if intent == "ask_timetable" and entities.get("destination") and not entities.get("towards"):
                    entities["towards"] = entities["destination"]
                    print(f"[NLP] Promoted destination='{entities['destination']}' → towards for timetable query")
                for k in ("origin", "destination", "origin_grounded", "destination_grounded"):
                    entities.pop(k, None)
        print(f"[NLP] Final intent: {intent}, entities: {entities}")
        # Final safety: if, after all filters, an origin/destination still starts
        # with a verb/pronoun due to some unexpected path, drop it here as well.
        for slot_key in ("origin", "destination"):
            val = (entities.get(slot_key) or "").strip()
            if not val:
                continue
            first_word = val.split()[0].lower().strip(" '\"“”‘’(),.")
            bad_starts = {
                "i", "i'm", "im", "me", "you", "we", "they", "he", "she", "it",
                "my", "your", "our", "their", "his", "her", "its",
                "this", "that", "these", "those",
                "go", "going", "get", "getting", "take", "taking",
                "plan", "planning", "travel", "travelling", "traveling",
                "leave", "leaving", "depart", "departing",
                "start", "starting", "head", "heading", "navigate",
                "navigating", "walk", "walking", "drive", "driving",
                "catch", "catching", "need", "needing", "want", "wanting",
                "know", "see", "make", "do", "be", "have",
            }
            if first_word in bad_starts:
                entities.pop(slot_key, None)

        return {
            'intent': intent,
            'entities': entities,
            'confidence': confidence
        }
    
    def _normalize_llm_entities(self, llm_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize LLM-extracted entities to match the key names expected
        by the rest of the pipeline (journey planner, chatbot, frontend).

        The LLM uses richer, more descriptive keys; this maps them to the
        existing entity schema while preserving the extra LLM-specific fields
        prefixed with 'llm_' for the frontend info panel.
        """
        entities: Dict[str, Any] = {}

        # Direct pass-through keys (same name in both schemas)
        # Note: "towards" here is the timetable-specific entity (bus travel direction),
        # distinct from journey planner origin/destination.  It must not overwrite
        # journey-planner slots and is only meaningful when intent == 'ask_timetable'.
        for key in ("origin", "destination", "location", "date", "time",
                     "bus_route", "near_area", "towards"):
            if key in llm_entities:
                entities[key] = llm_entities[key]

        # ── TfL Journey API parameters (new) ──

        # via — intermediate waypoint for the journey
        if "via" in llm_entities:
            entities["via"] = llm_entities["via"]

        # mode — comma-separated TfL transport modes (e.g. "tube,walking")
        if "mode" in llm_entities:
            entities["mode"] = llm_entities["mode"]

        # time_preference → maps to TfL "timeIs" parameter
        # LLM outputs "departing"/"arriving" → TfL expects "Departing"/"Arriving"
        if "time_preference" in llm_entities:
            tp = llm_entities["time_preference"].lower()
            entities["time_preference"] = tp
            # Also store the TfL-formatted version
            entities["timeIs"] = "Arriving" if tp == "arriving" else "Departing"

        # journey_preference → maps to TfL "journeyPreference" parameter
        # Values: "leastinterchange", "leasttime", "leastwalking"
        if "journey_preference" in llm_entities:
            entities["journey_preference"] = llm_entities["journey_preference"]

        # tube_line → line (and also route for compatibility)
        if "tube_line" in llm_entities:
            entities["line"] = llm_entities["tube_line"]
            entities["route"] = llm_entities["tube_line"]

        # Legacy travel_mode for timetable routing (bus/train detection)
        if "mode" in llm_entities:
            mode_str = llm_entities["mode"].lower()
            if "public-bus" in mode_str and "tube" not in mode_str and "train" not in mode_str:
                entities["timetable_mode"] = "bus"
                entities["travel_mode"] = "transit"
            elif ("tube" in mode_str or "train" in mode_str) and "public-bus" not in mode_str:
                entities["timetable_mode"] = "train"
                entities["travel_mode"] = "transit"
            elif "walking" == mode_str:
                entities["travel_mode"] = "walk"
            elif "cycle" == mode_str:
                entities["travel_mode"] = "bike"
            else:
                entities["travel_mode"] = "transit"

        # accessibility needs
        if "accessibility" in llm_entities:
            entities["accessibility"] = llm_entities["accessibility"]

        # ── LLM-enriched context fields (prefixed for the info panel) ──
        for key in ("urgency", "mood", "utterance_type"):
            if key in llm_entities:
                entities[f"llm_{key}"] = llm_entities[key]

        return entities

    def _classify_intent(self, text: str) -> tuple:
        """
        Classify user intent using a 3-tier strategy:
          1. High-confidence rule-based patterns (fast, catches unambiguous cases)
          2. Fine-tuned sentence-transformer classifier (main classifier)
          3. Rule-based fallback with synonym expansion (if classifier unavailable)
        """
        text_lower = text.lower()

        # ---- Tier 1: High-confidence rule-based overrides ----
        # These catch unambiguous patterns that should not go through ML
        if any(phrase in text_lower for phrase in [
            'bus times', 'train times', 'tube times', 'bus or train times',
            'train or bus times', 'timetable', 'next bus', 'next train',
            'when is the next',
        ]):
            return 'ask_timetable', 0.95

        _disruption_keywords = [
            'disruption', 'disrupted', 'status', 'delay', 'delays',
            'okay', 'ok', 'alright', 'fine', 'running', 'working',
            'problems', 'problem', 'issues', 'issue',
        ]
        if any(kw in text_lower for kw in _disruption_keywords):
            has_train = any(
                x in text_lower for x in [
                    ' line', ' tube', 'train', 'overground', 'dlr', 'underground',
                    'bakerloo', 'central', 'circle', 'district', 'hammersmith', 'jubilee',
                    'metropolitan', 'northern', 'piccadilly', 'victoria', 'waterloo',
                    'windrush', 'lioness', 'mildmay', 'suffragette', 'weaver', 'liberty',
                ]
            )
            has_bus_route = bool(re.search(r'bus.*\d|\d.*bus', text_lower))
            has_bus_status = 'bus' in text_lower
            if has_train or has_bus_route or has_bus_status:
                return 'ask_transit_disruption', 0.95

        # ---- Tier 2: Fine-tuned intent classifier ----
        if self._intent_classifier.is_ready:
            details = self._intent_classifier.classify_with_details(text)
            intent = details["intent"]
            confidence = details["confidence"]
            all_scores = details.get("all_scores", [])
            print(f"[NLP] Classifier intent: {intent}, confidence: {confidence:.3f}")

            # If the top-2 intents are very close, check rule-based for a tiebreaker
            if len(all_scores) >= 2:
                top_conf = all_scores[0][1]
                second_conf = all_scores[1][1]
                if top_conf < 1.5 * second_conf and top_conf < 0.7:
                    # Ambiguous — consult rule-based
                    rule_intent, rule_confidence = self._rule_based_intent(text)
                    if rule_confidence > 0.8:
                        print(f"[NLP] Ambiguous classifier; using rule-based: {rule_intent} ({rule_confidence:.3f})")
                        return rule_intent, rule_confidence

            # Raised threshold: only trust classifier if confidence > 0.5
            if confidence >= 0.5:
                return intent, confidence

            # Low confidence — fall back to rule-based
            rule_intent, rule_confidence = self._rule_based_intent(text)
            print(f"[NLP] Low classifier confidence ({confidence:.3f}); using rule-based: {rule_intent} ({rule_confidence:.3f})")
            return rule_intent, max(rule_confidence, confidence)

        # ---- Tier 3: Legacy zero-shot classifier (if new one not available) ----
        if self.intent_classifier:
            try:
                result = self.intent_classifier(text, self.intent_labels)
                intent = result['labels'][0]
                confidence = result['scores'][0]
                print(f"[NLP] Legacy zero-shot intent: {intent}, confidence: {confidence:.3f}")
                if confidence >= 0.6:
                    return intent, confidence
            except Exception as e:
                print(f"[NLP] Legacy classifier error: {e}")

        # ---- Tier 4: Pure rule-based fallback ----
        return self._rule_based_intent(text)
    
    def _rule_based_intent(self, text: str) -> tuple:
        """Rule-based intent detection as fallback - covers all intents A-K"""
        text_lower = text.lower()
        # Tokenize and expand with WordNet synonyms (if available)
        tokens = re.findall(r'\w+', text_lower)
        expanded_words = self._expand_with_synonyms(tokens)
        print(f"[NLP] Rule-based tokens: {tokens}")
        print(f"[NLP] Rule-based expanded_words: {sorted(expanded_words)}")

        # Concept sets for synonym-aware matching
        traffic_concepts = {'traffic', 'congestion', 'jam'}
        greeting_concepts = {'hello', 'hi', 'hey', 'greetings'}
        goodbye_concepts = {'bye', 'goodbye', 'farewell'}
        disruption_concepts = {'disruption', 'disturbance', 'interruption'}
        transit_concepts = {'transit', 'train', 'tube', 'subway', 'metro', 'tram', 'bus'}
        timetable_concepts = {'timetable', 'schedule', 'times', 'time'}
        delay_concepts = {'delay', 'late', 'slow', 'holdup', 'queue'}
        route_concepts = {'route', 'way', 'path', 'directions'}

        has_traffic_concept = bool(traffic_concepts & expanded_words)
        has_greeting_concept = bool(greeting_concepts & expanded_words)
        has_goodbye_concept = bool(goodbye_concepts & expanded_words)
        has_disruption_concept = bool(disruption_concepts & expanded_words)
        has_transit_concept = bool(transit_concepts & expanded_words)
        has_timetable_concept = bool(timetable_concepts & expanded_words)
        has_delay_concept = bool(delay_concepts & expanded_words)
        has_route_concept = bool(route_concepts & expanded_words)
        print(
            "[NLP] Concepts - "
            f"traffic={has_traffic_concept}, greeting={has_greeting_concept}, "
            f"goodbye={has_goodbye_concept}, disruption={has_disruption_concept}, "
            f"transit={has_transit_concept}, timetable={has_timetable_concept}, "
            f"delay={has_delay_concept}, route={has_route_concept}"
        )
        
        # Greeting patterns
        if has_greeting_concept or any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            print("[NLP] Rule-based matched: greeting")
            return 'greeting', 0.9
        
        # Goodbye patterns
        if has_goodbye_concept or any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            print("[NLP] Rule-based matched: goodbye")
            return 'goodbye', 0.9
        
        # G) Public transport disruption
        if (
            has_disruption_concept and has_transit_concept
        ) or any(
            phrase in text_lower
            for phrase in [
                'tube disruption', 'line disruption', 'train disruption',
                'transit disruption', 'bus disruption', 'disruption on the bus',
                'disruption on bus', 'disruption on the train', 'disruption on train',
                'disruption on the tube', 'disruption on tube', 'public transport'
            ]
        ):
            print("[NLP] Rule-based matched: ask_transit_disruption")
            return 'ask_transit_disruption', 0.85
        # Multimodal / comparative travel mode queries now map into journey planning
        if any(phrase in text_lower for phrase in ['faster than driving', 'public transport faster', 'multimodal', 'park and ride']):
            print("[NLP] Rule-based matched: journey_planning (from multimodal cue)")
            return 'journey_planning', 0.85

        # G2) Transit timetable/times - check early before generic traffic queries
        if has_timetable_concept and has_transit_concept or any(
            phrase in text_lower
            for phrase in [
                'bus times', 'train times', 'tube times', 'bus or train times',
                'train or bus times', 'timetable', 'next bus', 'next train', 'when is the next'
            ]
        ):
            print("[NLP] Rule-based matched: ask_timetable")
            return 'ask_timetable', 0.9
        
        # C) ETA / arrival time (previously separate delay intent)
        if has_delay_concept or any(
            phrase in text_lower
            for phrase in ['delay', 'how long', 'wait time', 'stuck', 'slow']
        ):
            # Route/transport delay questions are closest to disruption for the remaining intents
            print("[NLP] Rule-based matched: ask_transit_disruption (from delay cue)")
            return 'ask_transit_disruption', 0.85
        
        # A) Road traffic / congestion style queries no longer have dedicated intents.
        # We deliberately avoid mapping them to another supported intent so they
        # fall through to 'unknown' and are handled by the generic responder.
        if any(
            phrase in text_lower
            for phrase in ['what is the traffic like', 'what\'s the traffic like', 'traffic like in', 'traffic like on']
        ):
            print("[NLP] Rule-based matched: traffic-like query (mapped to unknown)")
            return 'unknown', 0.7
        if any(
            phrase in text_lower
            for phrase in ['how\'s traffic', 'traffic right now', 'traffic now', 'traffic status', 'traffic condition', 'congestion near']
        ) or has_traffic_concept:
            print("[NLP] Rule-based matched: traffic/congestion query (mapped to unknown)")
            return 'unknown', 0.7
        if 'is the' in text_lower and ('traffic' in text_lower or has_traffic_concept):
            print("[NLP] Rule-based matched: traffic \"is the\" query (mapped to unknown)")
            return 'unknown', 0.7
        
        return 'unknown', 0.5
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract entities like location, route, time, origin, destination
        
        Uses regex patterns and keyword matching
        """
        entities = {}
        text_lower = text.lower()
        
        # Timetable location ("times for X", "at X") is resolved via CSV fuzzy match in _refine_with_stop_datasets.
        # Extract location from "traffic like in [location]" or "traffic in [location]" patterns
        # This must come before generic location patterns to avoid capturing query words
        if 'location' not in entities:
            traffic_in_pattern = r'traffic\s+(?:like\s+)?in\s+([a-z]+(?:\s+[a-z]+)*)'
            match = re.search(traffic_in_pattern, text_lower)
            if match:
                location_raw = match.group(1).strip()
                # Filter out query words
                query_words = ['is', 'the', 'like', 'traffic', 'what', 'how', 'where']
                words = location_raw.split()
                cleaned_words = [w for w in words if w not in query_words]
                # Validate: must have at least one meaningful word
                if cleaned_words and len(' '.join(cleaned_words).strip()) >= 2:
                    cleaned_location = ' '.join(word.capitalize() for word in cleaned_words)
                    entities['location'] = cleaned_location
        
        # Also check for "traffic on [route]" patterns
        traffic_on_pattern = r'traffic\s+(?:like\s+)?on\s+([a-z]+(?:\s+[a-z]+)*(?:\s+(?:highway|road|street|avenue|way))?)'
        match = re.search(traffic_on_pattern, text_lower)
        if match and 'location' not in entities:
            route_raw = match.group(1).strip()
            query_words = ['is', 'the', 'like', 'traffic', 'what', 'how']
            words = route_raw.split()
            cleaned_words = [w for w in words if w not in query_words]
            # Validate: must have at least one meaningful word
            if cleaned_words and len(' '.join(cleaned_words).strip()) >= 2:
                cleaned_route = ' '.join(word.capitalize() for word in cleaned_words)
                entities['route'] = cleaned_route
        
        # Extract bus route numbers early (before generic location patterns)
        # This handles: "bus 83", "the bus 83", "route 83", "disruption on the bus 83"
        bus_route_patterns = [
            r'(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
            r'bus\s+route\s+(\d{1,3}|[Nn]\d{1,3})',
            # Allow "route 83" at end of phrase or before "bus"
            r'route\s+(\d{1,3}|[Nn]\d{1,3})(?:\s+bus)?\b',
            r'(?:disruption|status|delay)\s+(?:on\s+)?(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
        ]
        for pattern in bus_route_patterns:
            match = re.search(pattern, text_lower)
            if match:
                route_num = match.group(1).upper()
                # Set both route and line for disruption handler
                entities['route'] = route_num
                entities['line'] = route_num
                break
        
        # Extract location/route (common patterns)
        # Look for phrases like "on Highway 101", "in downtown", "to airport"
        # Also recognize London transport lines and stations
        location_patterns = [
            r'(?:on|in|at|near|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Capitalized place names
            r'(?:on|in|at|near|to|from)\s+([a-z]+\s+(?:street|road|avenue|highway|freeway|boulevard|way))',
            r'highway\s+(\d+)',
            r'route\s+(\d+)',
            r'hwy\s+(\d+)',
            # London transport patterns
            r'(?:the\s+)?(bakerloo|central|circle|district|hammersmith|jubilee|metropolitan|northern|piccadilly|victoria|waterloo|dlr|overground|tram)\s+(?:line|tube)',
            r'(?:line|tube)\s+(bakerloo|central|circle|district|hammersmith|jubilee|metropolitan|northern|piccadilly|victoria|waterloo)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                match_value = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
                # Clean up: remove common query words
                query_words = ['is', 'the', 'like', 'traffic', 'what', 'how', 'where']
                words = match_value.split()
                cleaned_words = [w for w in words if w.lower() not in query_words]
                if not cleaned_words:
                    continue
                match_value = ' '.join(cleaned_words)
                
                # Skip if this is "bus" and we're in a disruption query (bus route will be extracted separately)
                if match_value.lower() == 'bus' and ('disruption' in text_lower or 'status' in text_lower):
                    continue
                
                # Check if it's a known train line (from _train_lines)
                match_norm = self._normalize_for_line_match(match_value)
                is_train_line = any(
                    self._normalize_for_line_match(ln) in match_norm or match_norm in self._normalize_for_line_match(ln)
                    for ln in self._train_lines
                )
                
                # Don't extract as route if it's a timetable query - we already have the location
                if ('times' in text_lower or 'timetable' in text_lower) and 'location' in entities:
                    # This is a timetable query, skip route extraction entirely
                    continue
                
                # Don't extract as route if the match contains "for" and we're in a timetable query
                if ('times' in text_lower or 'timetable' in text_lower) and 'for' in match_value.lower():
                    # This is likely "times for [location]", don't extract as route
                    continue
                
                if is_train_line:
                    if 'route' not in entities:
                        entities['route'] = match_value
                else:
                    # Only set location for place/street names; do not set route
                    # (route is set by bus_route_patterns or tube line patterns above)
                    # Skip regex location for timetable-like queries (for/near/from/to etc.) so CSV method captures from full text
                    is_timetable_query = any(
                        phrase in text_lower for phrase in [
                            'times', 'timetable', 'next bus', 'next train', 'next tube',
                            'bus times', 'train times', 'tube times'
                        ]
                    )
                    if 'location' not in entities and not is_timetable_query:
                        entities['location'] = match_value
        
        # Extract time
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(?:am|pm)?',
            r'(\d{1,2})\s*(?:am|pm)',
            r'(morning|afternoon|evening|night|noon|midnight)',
            r'(today|tomorrow|now|later)',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                entities['time'] = match.group(0)
                break
        
        # Origin and destination extraction is handled exclusively by the
        # LLM-based extractor (llm_entity_extractor.py). No rule-based
        # journey slot extraction is performed here.

        # Extract travel mode
        if re.search(r'\b(drive|driving|car)\b', text_lower):
            entities['travel_mode'] = 'drive'
        elif re.search(r'\b(transit|train|tube|bus|public transport|transport)\b', text_lower):
            entities['travel_mode'] = 'transit'
        elif re.search(r'\b(bike|bicycle|cycling)\b', text_lower):
            entities['travel_mode'] = 'bike'
        elif re.search(r'\b(walk|walking|on foot)\b', text_lower):
            entities['travel_mode'] = 'walk'
        
        # Extract timetable mode preference (bus, train, or both)
        if any(phrase in text_lower for phrase in ['bus times', 'bus time', 'next bus']):
            if 'train' not in text_lower and 'tube' not in text_lower:
                entities['timetable_mode'] = 'bus'
            else:
                entities['timetable_mode'] = 'both'
        elif any(phrase in text_lower for phrase in ['train times', 'train time', 'tube times', 'tube time', 'next train', 'next tube']):
            if 'bus' not in text_lower:
                entities['timetable_mode'] = 'train'
            else:
                entities['timetable_mode'] = 'both'
        elif any(phrase in text_lower for phrase in ['bus or train times', 'train or bus times', 'bus and train times', 'timetable']):
            entities['timetable_mode'] = 'both'
        
        # Extract preferences
        if re.search(r'\b(avoid tolls|no tolls|without tolls)\b', text_lower):
            entities['avoid_tolls'] = True
        if re.search(r'\b(avoid motorways|no motorways|without motorways)\b', text_lower):
            entities['avoid_motorways'] = True
        if re.search(r'\b(avoid ferries|no ferries)\b', text_lower):
            entities['avoid_ferries'] = True
        
        # Extract ULEZ/LEZ/Congestion charge queries
        if re.search(r'\b(ulez|lez|congestion charge)\b', text_lower):
            entities['ulez'] = True
            entities['congestion_charge'] = True
        
        # Extract avoid area
        avoid_pattern = r'avoid\s+(?:this\s+)?area[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(avoid_pattern, text, re.IGNORECASE)
        if match:
            entities['avoid_area'] = match.group(1).strip()
        
        # Extract event information
        event_pattern = r'(?:going to|match at|event at|concert at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(event_pattern, text, re.IGNORECASE)
        if match:
            entities['event'] = match.group(1).strip()
        
        # Extract line/route for transit
        line_pattern = r'(?:the\s+)?(central\s+line|bakerloo\s+line|northern\s+line|piccadilly\s+line|etc\.?)'
        match = re.search(line_pattern, text_lower)
        if match:
            entities['line'] = match.group(1).strip()
        
        # Extract road names for incidents
        road_pattern = r'\b(A\d+|M\d+|M\d+[A-Z]|Junction\s+\d+)\b'
        match = re.search(road_pattern, text, re.IGNORECASE)
        if match:
            entities['road'] = match.group(1).strip()
        
        # Extract common location keywords
        common_locations = {
            'downtown': 'downtown',
            'airport': 'airport',
            'city center': 'city center',
            'suburb': 'suburbs',
            'highway': 'highway',
            'freeway': 'freeway',
            # London stations
            'oxford circus': 'Oxford Circus',
            'kings cross': "King's Cross",
            'paddington': 'Paddington',
            'victoria': 'Victoria',
            'waterloo': 'Waterloo',
            'liverpool street': 'Liverpool Street',
            'euston': 'Euston',
            'st pancras': "St Pancras",
            'charing cross': 'Charing Cross',
            'piccadilly circus': 'Piccadilly Circus',
        }
        
        for keyword, location in common_locations.items():
            if keyword in text_lower and 'location' not in entities:
                entities['location'] = location
                break
        
        return entities
    
    def _load_stop_datasets(self) -> None:
        """
        Legacy method — stop data is loaded by `tfl_stop_datasets.py`.
        This stub ensures backward compatibility if called externally.
        """
        if self._stops_loaded:
            return
        self._bus_stops = load_bus_stops()
        self._train_stations = load_train_stations()
        self._train_lines = list(TRAIN_LINES)
        self._bus_routes = load_bus_routes()
        self._stops_loaded = True
    
    # Keywords that indicate a status/disruption query (train or bus)
    _DISRUPTION_KEYWORDS = (
        'status', 'disruption', 'disrupted', 'delay', 'delays', 'closure', 'closed',
        'problem', 'problems', 'issue', 'issues', 'service', 'running', 'working',
        'okay', 'ok', 'alright', 'fine', 'good', 'behaving',
    )
    
    def _normalize_for_line_match(self, text: str) -> str:
        """Normalize text for train line matching: lowercase, & interchangeable with 'and', apostrophes optional."""
        t = text.lower().strip()
        t = re.sub(r"['\u2019]", '', t)  # remove apostrophes
        t = re.sub(r'\s*&\s*', ' and ', t)
        t = re.sub(r'\s+', ' ', t)
        return t
    
    def extract_train_disruption_line(self, query: str) -> Optional[str]:
        """
        If the query is about train/tube/Overground/DLR status or disruption and mentions
        a line from _train_lines, return that line's display name. Otherwise return None.
        Matching: punctuation like & is interchangeable; apostrophes don't have to be included.
        """
        if not query or not self._train_lines:
            return None
        q = self._normalize_for_line_match(query)
        if not any(kw in q for kw in self._DISRUPTION_KEYWORDS):
            return None

        # First try exact/substring matching, then fall back to fuzzy similarity
        best_name: Optional[str] = None
        best_score: float = 0.0

        for line_name in self._train_lines:
            norm_line = self._normalize_for_line_match(line_name)
            if not norm_line:
                continue
            # Strong match when the normalized line name appears directly in the query
            if norm_line in q:
                return line_name
            # Fuzzy fallback: allow small typos like "bakerlo" for "bakerloo"
            score = SequenceMatcher(None, q, norm_line).ratio()
            if score > best_score:
                best_score = score
                best_name = line_name

        # Accept moderately strong fuzzy matches (small spelling mistakes)
        if best_name is not None and best_score >= 0.80:
            return best_name
        return None
    
    def extract_bus_disruption_route(self, query: str) -> Optional[str]:
        """
        If the query is about bus status or disruption and contains a route number/string
        that appears in tfl_bus_routes.txt, return that route id (e.g. "83", "N29"). Otherwise return None.
        """
        if not query or not self._bus_routes:
            return None
        q_lower = query.lower()
        if not any(kw in q_lower for kw in self._DISRUPTION_KEYWORDS):
            return None
        # Find candidate route tokens: N? digits (e.g. 83, N29)
        candidates = re.findall(r'\b([Nn]?\d{1,3})\b', query)
        for c in candidates:
            if c.upper() in self._bus_routes or c in self._bus_routes:
                return c.upper() if c.upper() in self._bus_routes else c
        return None
    
    def parse_line_or_route_followup(self, message: str) -> Optional[Tuple[str, str]]:
        """
        For follow-up replies after "couldn't find a train/bus": if the message is just a train line
        or bus route (no status/disruption keywords required), return (value, 'train') or (value, 'bus').
        Otherwise return None. Bus route is preferred when the message is only digits or N+digits.
        """
        if not message or not message.strip():
            return None
        msg = message.strip()
        q = self._normalize_for_line_match(msg)
        # Bus: single token that is in _bus_routes (e.g. "83", "N29")
        bus_candidates = re.findall(r'\b([Nn]?\d{1,3})\b', msg)
        if len(bus_candidates) == 1 and (bus_candidates[0].upper() in self._bus_routes or bus_candidates[0] in self._bus_routes):
            return (bus_candidates[0].upper() if bus_candidates[0].upper() in self._bus_routes else bus_candidates[0], 'bus')
        # Train: message matches, is contained in, or is a close fuzzy match to a _train_lines name
        best_name: Optional[str] = None
        best_score: float = 0.0
        for line_name in self._train_lines:
            norm_line = self._normalize_for_line_match(line_name)
            if not norm_line:
                continue
            # Exact / substring matches are preferred
            if norm_line == q or norm_line in q or (q in norm_line and len(q) >= 3):
                return (line_name, 'train')
            # Fuzzy fallback: tolerate minor typos in short replies like "victora"
            score = SequenceMatcher(None, q, norm_line).ratio()
            if score > best_score:
                best_score = score
                best_name = line_name

        if best_name is not None and best_score >= 0.80:
            return (best_name, 'train')
        return None
    
    def _best_csv_stop_match(self, candidate: str, mode_hint: Optional[str] = None):
        """
        Given a free-text candidate (usually a location phrase), find the best
        fuzzy match across bus and/or train CSV stop names.
        Returns a dict with keys: type ('bus'|'train'), name, score, or None.

        mode_hint: If "train", only match against train stations (so train
        queries don't match bus CSV). If "bus", only match against bus stops.
        If None, search both and return the single best score (previous behaviour).
        """
        if not candidate or not candidate.strip():
            return None
        
        if not self._stops_loaded:
            self._load_stop_datasets()
        
        cand = candidate.strip().lower()
        if not cand:
            return None
        # Avoid mapping very short, single-word phrases like "home" or "uni"
        # directly to a random stop via fuzzy matching. These are typically
        # handled via user-defined shortcuts upstream; if no shortcut was
        # applied, it's safer to leave them as-is.
        cand_tokens = cand.split()
        if len(cand_tokens) == 1 and len(cand_tokens[0]) <= 4:
            return None
        
        best_type = None
        best_name = None
        best_score = 0.0
        
        # Helper: score a stop name against the candidate, preferring
        # exact substring containment over general fuzziness.
        def _score_stop(stop_name: str) -> float:
            stop_lower = stop_name.lower()
            # If the full stop name appears inside the candidate text,
            # treat this as a near-perfect match even if there are extra words.
            if stop_lower and stop_lower in cand:
                return 1.0
            # If the candidate appears inside the stop name (short query like "Baker"),
            # still give a strong score.
            if cand and cand in stop_lower:
                return 0.95
            # Fallback to character-based similarity for typos / small deviations.
            return SequenceMatcher(None, cand, stop_lower).ratio()
        
        search_bus = mode_hint != "train"
        search_train = mode_hint != "bus"

        if search_bus:
            for name in self._bus_stops:
                s = _score_stop(name)
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_type = "bus"

        if search_train:
            for name in self._train_stations:
                s = _score_stop(name)
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_type = "train"
        
        if best_name is None or best_type is None:
            return None
        
        return {"type": best_type, "name": best_name, "score": best_score}
    
    def _refine_with_stop_datasets(
        self,
        original_text: str,
        intent: str,
        entities: Dict[str, Any],
        confidence: float,
    ):
        """
        Use CSV stop names as an additional NER + slot-filling and intent hint layer.
        
        - Detect if the user text closely matches a known bus or train stop.
        - Infer whether this is a bus or train timetable query.
        - Cooperate with existing rule-based / transformer intent detection and
          with the downstream disambiguation engine in the TFL API.
        """
        text_lower = original_text.lower()

        # Only run this refinement when the user is clearly talking about
        # buses/trains (regardless of whether they say "times"/"timetable").
        has_bus_words = any(w in text_lower for w in ["bus", "coach"])
        has_train_words = any(
            w in text_lower
            for w in ["train", "tube", "rail", "overground", "dlr", "underground"]
        )
        if not (has_bus_words or has_train_words):
            return intent, entities, confidence

        # When user said only train (or only bus), search that CSV only so we
        # don't match the same place name to the other mode (e.g. "train times
        # for Baker Street" must match train_stops.csv, not bus_stops.csv).
        mode_hint = None
        if has_train_words and not has_bus_words:
            mode_hint = "train"
        elif has_bus_words and not has_train_words:
            mode_hint = "bus"

        # Prefer an already extracted location; otherwise fall back to the whole text
        candidate = entities.get("location") or original_text
        match = self._best_csv_stop_match(candidate, mode_hint=mode_hint)
        # Accept moderately strong fuzzy matches so that small typos / variations
        # in stop names are still recognized, without being overly permissive.
        if not match or match.get("score", 0.0) < 0.80:
            # No strong stop-name match; keep existing interpretation
            return intent, entities, confidence
        
        stop_type = match["type"]
        stop_name = match["name"]
        score = match.get("score", 0.0)
        print(f"[NLP][CSV] Matched stop '{stop_name}' (type={stop_type}, score={score:.3f}) from CSVs")
        
        current_mode = entities.get("timetable_mode")
        inferred_mode = None
        
        # Decide timetable mode primarily from the CSV type,
        # but respect an existing compatible mode if already set.
        if stop_type == "bus":
            if current_mode in (None, "bus", "both"):
                inferred_mode = "bus"
            else:
                inferred_mode = current_mode
        elif stop_type == "train":
            if current_mode in (None, "train", "both"):
                inferred_mode = "train"
            else:
                inferred_mode = current_mode
        
        if not inferred_mode:
            return intent, entities, confidence
        
        # Fill/normalize NER slots for downstream dialogue state tracking.
        # IMPORTANT: we *do not* overwrite the user's original location phrase
        # if we already have one (e.g. "oxford street", "high road"). Instead:
        # - keep the user's extracted phrase in entities["location"] so prompts
        #   and TfL queries use exactly what they typed
        # - store the CSV-backed canonical stop name separately so downstream
        #   code can still use it if needed.
        #
        # Only when there was no prior location extracted do we fall back to
        # using the canonical stop name as the location.
        display_name = stop_name
        candidate_lower = (candidate or "").lower()
        stop_lower = stop_name.lower()
        if stop_type == "bus":
            if "station" in candidate_lower and "station" not in stop_lower:
                display_name = f"{stop_name} Station"
        elif stop_type == "train":
            # Keep "underground station" or "station" in the query for TfL API
            if "underground station" in candidate_lower and "underground station" not in stop_lower:
                display_name = f"{stop_name} Underground Station"
            elif "station" in candidate_lower and "station" not in stop_lower:
                display_name = f"{stop_name} Station"

        existing_location = (entities.get("location") or "").strip()
        if not existing_location:
            # No location was previously extracted – use the canonical stop name.
            entities["location"] = display_name
        else:
            # Preserve the user's phrase and keep the CSV stop separately.
            entities["csv_stop_name"] = display_name

        entities["timetable_mode"] = inferred_mode
        entities["timetable_stop_source"] = f"{stop_type}_csv"
        print(
            f"[NLP][CSV] Using '{stop_name}' as location with timetable_mode='{inferred_mode}' "
            f"(source={stop_type}_csv)"
        )
        
        # For queries that mention bus/train plus a recognized stop, we want to
        # take the same path as "bus times for ..." / "train times for ...".
        # So, unless the user is clearly asking about disruption rather than times,
        # treat this as a timetable query.
        if "disruption" not in text_lower and "status" not in text_lower:
            intent = "ask_timetable"
            confidence = max(confidence, 0.9)
        
        return intent, entities, confidence
