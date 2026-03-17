"""
Hybrid NER Processor: SpaCy + Domain-Specific Regex.

Combines:
  1. SpaCy's pre-trained NER for general entities (locations, dates, times, orgs)
  2. Domain-specific regex patterns for TfL-specific entities (bus routes, tube lines, etc.)

The SpaCy model catches natural language entity mentions the regex would miss,
while the regex ensures we never miss domain-specific patterns (route numbers,
line names, etc.) that SpaCy hasn't been trained on.
"""

import re
import os
import csv
from typing import Dict, Any, List, Set, Optional, Tuple
from difflib import SequenceMatcher

from journey_slot_extractor import get_extractor as _get_journey_extractor


class NERProcessor:
    """
    Hybrid Named Entity Recognition combining SpaCy NER with transport-domain regex.
    """

    def __init__(self):
        """Initialize SpaCy model and domain knowledge."""
        self._spacy_nlp = None
        self._spacy_available = False

        # Domain knowledge loaded from CSVs
        self._bus_stops: List[str] = []
        self._train_stations: List[str] = []
        self._train_lines: List[str] = []
        self._bus_routes: Set[str] = set()
        self._stops_loaded = False

        self._load_spacy()
        self._load_domain_data()

    def _load_spacy(self) -> None:
        """Load SpaCy model for general NER."""
        try:
            import spacy
            # Try the transformer-based model first, then medium, then small
            for model_name in ["en_core_web_trf", "en_core_web_md", "en_core_web_sm"]:
                try:
                    self._spacy_nlp = spacy.load(model_name)
                    print(f"[NER] Loaded SpaCy model: {model_name}")
                    self._spacy_available = True
                    return
                except OSError:
                    continue

            # If no model installed, try downloading the small one
            print("[NER] No SpaCy model found, attempting to download en_core_web_sm...")
            try:
                spacy.cli.download("en_core_web_sm")
                self._spacy_nlp = spacy.load("en_core_web_sm")
                self._spacy_available = True
                print("[NER] Successfully downloaded and loaded en_core_web_sm")
            except Exception as e:
                print(f"[NER] Could not download SpaCy model: {e}")
                print("[NER] SpaCy NER unavailable; using regex-only fallback.")

        except ImportError:
            print("[NER] SpaCy not installed. Install with: pip install spacy")
            print("[NER] Using regex-only NER fallback.")

    def _load_domain_data(self) -> None:
        """Load TfL-specific domain knowledge from CSVs."""
        if self._stops_loaded:
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        bus_path = os.path.join(base_dir, "bus_stops.csv")
        train_path = os.path.join(base_dir, "train_stops.csv")
        bus_routes_path = os.path.join(base_dir, "tfl_bus_routes.txt")

        # Load bus stops
        bus_names: List[str] = []
        try:
            if os.path.exists(bus_path):
                encoding = "utf-8-sig"
                try:
                    with open(bus_path, newline="", encoding=encoding) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            raw_name = (row.get("CommonName") or "").strip()
                            if raw_name:
                                name = re.sub(r"\s*\([^)]*\)\s*$", "", raw_name).strip()
                                if name:
                                    bus_names.append(name)
                except UnicodeDecodeError:
                    with open(bus_path, newline="", encoding="cp1252") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            raw_name = (row.get("CommonName") or "").strip()
                            if raw_name:
                                name = re.sub(r"\s*\([^)]*\)\s*$", "", raw_name).strip()
                                if name:
                                    bus_names.append(name)
        except Exception as e:
            print(f"[NER] Failed to load bus_stops.csv: {e}")

        # Load train stations
        train_names: List[str] = []
        try:
            if os.path.exists(train_path):
                def _get_station_name(row: dict) -> str:
                    raw = (row.get("Station") or row.get("Stop") or row.get("Name") or "").strip()
                    return re.sub(r"\s*\([^)]*\)\s*$", "", raw).strip() if raw else ""

                try:
                    with open(train_path, newline="", encoding="utf-8-sig") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            name = _get_station_name(row)
                            if name:
                                train_names.append(name)
                except UnicodeDecodeError:
                    with open(train_path, newline="", encoding="cp1252") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            name = _get_station_name(row)
                            if name:
                                train_names.append(name)
                train_names = list(dict.fromkeys(train_names))
        except Exception as e:
            print(f"[NER] Failed to load train_stops.csv: {e}")

        # Load bus route IDs
        bus_route_ids: Set[str] = set()
        try:
            if os.path.exists(bus_routes_path):
                with open(bus_routes_path, encoding="utf-8") as f:
                    for line in f:
                        rid = line.strip()
                        if rid:
                            bus_route_ids.add(rid)
                            bus_route_ids.add(rid.upper())
        except Exception as e:
            print(f"[NER] Failed to load tfl_bus_routes.txt: {e}")

        self._bus_stops = bus_names
        self._train_stations = train_names
        self._bus_routes = bus_route_ids

        # London Underground, Overground, and DLR line names
        self._train_lines = [
            "Bakerloo", "Central", "Circle", "District",
            "Hammersmith & City", "Jubilee", "Metropolitan",
            "Northern", "Piccadilly", "Victoria", "Waterloo & City",
            "London Overground", "Windrush", "Lioness", "Mildmay",
            "Suffragette", "Weaver", "Liberty",
            "DLR", "Docklands Light Railway", "Elizabeth",
        ]

        self._stops_loaded = True
        print(
            f"[NER] Loaded {len(self._bus_stops)} bus stops, "
            f"{len(self._train_stations)} train stations, "
            f"{len(self._bus_routes)} bus routes"
        )

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from user text using both SpaCy and regex.

        Returns a dict which may contain any of:
          - location: General location name
          - route: Bus route number or road name
          - line: Train/tube line name
          - time: Time reference
          - origin: Journey origin
          - destination: Journey destination
          - travel_mode: drive/transit/bike/walk
          - timetable_mode: bus/train/both
          - bus_route: Specific bus route number
          - road: Road identifier (A-roads, M-roads)
          - event: Event name
          - date: Date reference
          - spacy_locations: List of locations found by SpaCy
          - spacy_times: List of time expressions found by SpaCy
          - spacy_orgs: List of organizations found by SpaCy
        """
        entities: Dict[str, Any] = {}
        text_lower = text.lower().strip()

        # ---- Phase 1: SpaCy NER (general entities) ----
        spacy_entities = self._extract_spacy_entities(text)
        if spacy_entities:
            entities.update(spacy_entities)

        # ---- Phase 2: Journey origin/destination via rule-based slot extractor ----
        # JourneySlotExtractor uses ordered grammar rules (from X to Y, to Y from X,
        # between X and Y, leaving from X, arriving at Y, etc.) and optionally
        # grounds results via Google Places API.
        #
        # We apply an additional filter so that origin/destination phrases never
        # start with verbs or pronouns and do not contain internal verbs (to avoid
        # extracting control phrases like "plan a journey" as destinations).
        try:
            journey_extractor = _get_journey_extractor()
            if journey_extractor.ready:
                slots = journey_extractor.extract_journey_slots(text)

                def _slot_is_valid_place_phrase(phrase: Optional[str]) -> bool:
                    """
                    Return True if *phrase* looks like a place name rather than a control phrase.
                    Rules:
                      - if it starts with a verb or pronoun → reject
                      - if it contains any verb (non-initial) → reject (checked via SpaCy POS when available)
                    """
                    if not phrase:
                        return False
                    raw = phrase.strip()
                    if not raw:
                        return False
                    first_word = raw.split()[0].lower().strip(" '\"“”‘’(),.")
                    bad_starts = {
                        # Pronouns / determiners
                        "i", "i'm", "im", "me", "you", "we", "they", "he", "she", "it",
                        "my", "your", "our", "their", "his", "her", "its",
                        "this", "that", "these", "those",
                        # Common journey verbs
                        "go", "going", "get", "getting", "take", "taking",
                        "plan", "planning", "travel", "travelling", "traveling",
                        "leave", "leaving", "depart", "departing",
                        "start", "starting", "head", "heading", "navigate",
                        "navigating", "walk", "walking", "drive", "driving",
                        "catch", "catching", "need", "needing", "want", "wanting",
                        "know", "see", "make", "do", "be", "have",
                    }
                    if first_word in bad_starts:
                        return False

                    # Use SpaCy POS tagging when available to detect internal verbs.
                    if self._spacy_available and self._spacy_nlp is not None:
                        try:
                            doc = self._spacy_nlp(raw)
                            for i, token in enumerate(doc):
                                # If the first token is VERB/AUX/PRON, we already rejected via bad_starts.
                                if i == 0:
                                    continue
                                if token.pos_ in ("VERB", "AUX"):
                                    return False
                        except Exception:
                            # Fall back silently if SpaCy fails.
                            pass

                    return True

                origin = slots.get("origin")
                if origin and _slot_is_valid_place_phrase(origin):
                    entities["origin"] = origin
                    # If Places grounding resolved a canonical name, expose it too
                    if slots.get("origin_grounded"):
                        entities["origin_grounded"] = slots["origin_grounded"]

                destination = slots.get("destination")
                if destination and _slot_is_valid_place_phrase(destination):
                    entities["destination"] = destination
                    if slots.get("destination_grounded"):
                        entities["destination_grounded"] = slots["destination_grounded"]

        except Exception as exc:
            print(f"[NER] Journey slot extraction error: {exc}")

        # ---- Phase 3: Domain-specific regex ----
        domain_entities = self._extract_domain_entities(text, text_lower)
        # Domain entities take precedence for domain-specific fields
        for key, value in domain_entities.items():
            if key not in entities or key in (
                "route", "line", "bus_route", "timetable_mode",
                "travel_mode", "road",
            ):
                entities[key] = value

        # ---- Phase 3: Reconcile SpaCy + regex locations ----
        self._reconcile_locations(entities, text_lower)

        return entities

    def _extract_spacy_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using SpaCy NER."""
        if not self._spacy_available or self._spacy_nlp is None:
            return {}

        doc = self._spacy_nlp(text)
        entities: Dict[str, Any] = {}

        locations = []
        times = []
        orgs = []
        dates = []

        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                # Geographic/location entities
                locations.append(ent.text)
            elif ent.label_ == "TIME":
                times.append(ent.text)
            elif ent.label_ == "DATE":
                dates.append(ent.text)
            elif ent.label_ == "ORG":
                orgs.append(ent.text)

        if locations:
            entities["spacy_locations"] = locations
            # Use the first location as the primary location if no regex location found
            if len(locations) == 1:
                entities["location"] = locations[0]

        if times:
            entities["spacy_times"] = times
            if "time" not in entities:
                entities["time"] = times[0]

        if dates:
            entities["spacy_dates"] = dates
            if "date" not in entities:
                entities["date"] = dates[0]

        if orgs:
            entities["spacy_orgs"] = orgs

        return entities

    def _extract_domain_entities(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Extract transport-domain entities using regex patterns."""
        entities: Dict[str, Any] = {}

        # ---- Bus route numbers ----
        bus_route_patterns = [
            r'(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
            r'bus\s+route\s+(\d{1,3}|[Nn]\d{1,3})',
            r'route\s+(\d{1,3}|[Nn]\d{1,3})(?:\s+bus)?\b',
            r'(?:disruption|status|delay)\s+(?:on\s+)?(?:the\s+)?bus\s+(\d{1,3}|[Nn]\d{1,3})',
        ]
        for pattern in bus_route_patterns:
            match = re.search(pattern, text_lower)
            if match:
                route_num = match.group(1).upper()
                entities["route"] = route_num
                entities["line"] = route_num
                entities["bus_route"] = route_num
                break

        # ---- Train/tube line names ----
        line_patterns = [
            r'(?:the\s+)?(bakerloo|central|circle|district|hammersmith(?:\s*(?:&|and)\s*city)?|jubilee|metropolitan|northern|piccadilly|victoria|waterloo(?:\s*(?:&|and)\s*city)?|elizabeth|dlr|overground|windrush|lioness|mildmay|suffragette|weaver|liberty)\s*(?:line|tube)?',
            r'(?:line|tube)\s+(bakerloo|central|circle|district|hammersmith|jubilee|metropolitan|northern|piccadilly|victoria|waterloo|elizabeth)',
        ]
        for pattern in line_patterns:
            match = re.search(pattern, text_lower)
            if match:
                line_name = match.group(1).strip().title()
                if "line" not in entities:
                    entities["line"] = line_name
                if "route" not in entities:
                    entities["route"] = line_name
                break

        # ---- Traffic location patterns ----
        # "traffic in [location]" or "traffic like in [location]"
        traffic_loc = re.search(r'traffic\s+(?:like\s+)?(?:in|on)\s+([a-z]+(?:\s+[a-z]+)*)', text_lower)
        if traffic_loc:
            loc_raw = traffic_loc.group(1).strip()
            query_words = {'is', 'the', 'like', 'traffic', 'what', 'how', 'where'}
            cleaned = [w for w in loc_raw.split() if w not in query_words]
            if cleaned and len(' '.join(cleaned)) >= 2:
                entities["location"] = ' '.join(w.capitalize() for w in cleaned)

        # ---- Road identifiers ----
        road_match = re.search(r'\b(A\d+|M\d+[A-Z]?|Junction\s+\d+)\b', text, re.IGNORECASE)
        if road_match:
            entities["road"] = road_match.group(1).strip()

        # ---- Origin and destination ----
        # Handled by the JourneySlotExtractor (Phase 2) which uses
        # preposition-context extraction.  The old regex here was too
        # aggressive (e.g. "want to plan" → origin="want", dest="plan")
        # so it has been removed to avoid overwriting correct Phase-2
        # results with garbage.

        # ---- Time expressions ----
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(?:am|pm)?',
            r'(\d{1,2})\s*(?:am|pm)',
            r'(morning|afternoon|evening|night|noon|midnight)',
            r'(today|tomorrow|now|later)',
        ]
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if "time" not in entities:
                    entities["time"] = match.group(0)
                break

        # ---- Travel mode ----
        if re.search(r'\b(drive|driving|car)\b', text_lower):
            entities["travel_mode"] = "drive"
        elif re.search(r'\b(transit|train|tube|bus|public transport|transport)\b', text_lower):
            entities["travel_mode"] = "transit"
        elif re.search(r'\b(bike|bicycle|cycling)\b', text_lower):
            entities["travel_mode"] = "bike"
        elif re.search(r'\b(walk|walking|on foot)\b', text_lower):
            entities["travel_mode"] = "walk"

        # ---- Timetable mode preference ----
        if any(p in text_lower for p in ['bus times', 'bus time', 'next bus']):
            if 'train' not in text_lower and 'tube' not in text_lower:
                entities["timetable_mode"] = "bus"
            else:
                entities["timetable_mode"] = "both"
        elif any(p in text_lower for p in ['train times', 'train time', 'tube times', 'tube time', 'next train', 'next tube']):
            if 'bus' not in text_lower:
                entities["timetable_mode"] = "train"
            else:
                entities["timetable_mode"] = "both"
        elif any(p in text_lower for p in ['bus or train times', 'train or bus times', 'bus and train times', 'timetable']):
            entities["timetable_mode"] = "both"

        # ---- Preferences ----
        if re.search(r'\b(avoid tolls|no tolls|without tolls)\b', text_lower):
            entities["avoid_tolls"] = True
        if re.search(r'\b(avoid motorways|no motorways|without motorways)\b', text_lower):
            entities["avoid_motorways"] = True
        if re.search(r'\b(ulez|lez|congestion charge)\b', text_lower):
            entities["ulez"] = True
            entities["congestion_charge"] = True

        # ---- Event ----
        event_match = re.search(r'(?:going to|match at|event at|concert at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text, re.IGNORECASE)
        if event_match:
            entities["event"] = event_match.group(1).strip()

        return entities

    def _reconcile_locations(self, entities: Dict[str, Any], text_lower: str) -> None:
        """
        Reconcile locations from SpaCy and regex, preferring the most specific match.
        Also uses CSV-backed fuzzy matching to validate locations as transport stops.
        """
        spacy_locs = entities.pop("spacy_locations", [])
        spacy_times = entities.pop("spacy_times", [])
        spacy_dates = entities.pop("spacy_dates", [])
        spacy_orgs = entities.pop("spacy_orgs", [])

        # If we don't have a location from regex, try SpaCy locations
        if "location" not in entities and spacy_locs:
            # Check if any SpaCy location matches a known stop
            for loc in spacy_locs:
                if self._is_known_stop(loc):
                    entities["location"] = loc
                    break
            # If none matched a known stop, use the first SpaCy location
            if "location" not in entities:
                entities["location"] = spacy_locs[0]

        # Store SpaCy metadata for downstream use
        if spacy_locs:
            entities["_spacy_locations"] = spacy_locs
        if spacy_times:
            entities["_spacy_times"] = spacy_times
        if spacy_dates:
            entities["_spacy_dates"] = spacy_dates

    def _is_known_stop(self, name: str) -> bool:
        """Check if a name matches a known bus stop or train station."""
        name_lower = name.lower().strip()
        for stop in self._bus_stops:
            if name_lower in stop.lower() or stop.lower() in name_lower:
                return True
        for station in self._train_stations:
            if name_lower in station.lower() or station.lower() in name_lower:
                return True
        return False

    def fuzzy_match_stop(self, query: str, mode: Optional[str] = None) -> Optional[str]:
        """
        Fuzzy-match a query against known stops/stations.

        Args:
            query: User's location query
            mode: 'bus', 'train', or None for both

        Returns:
            Best matching stop name, or None
        """
        if not query or not query.strip():
            return None

        query_lower = query.lower().strip()
        stops = []
        if mode in (None, "bus"):
            stops.extend(self._bus_stops)
        if mode in (None, "train"):
            stops.extend(self._train_stations)

        if not stops:
            return None

        best_match = None
        best_score = 0.0
        for stop in stops:
            stop_lower = stop.lower()
            # Exact substring match
            if query_lower in stop_lower or stop_lower in query_lower:
                score = 0.9
            else:
                score = SequenceMatcher(None, query_lower, stop_lower).ratio()

            if score > best_score:
                best_score = score
                best_match = stop

        return best_match if best_score > 0.4 else None

    @property
    def train_lines(self) -> List[str]:
        """Return known train/tube line names."""
        return self._train_lines

    @property
    def bus_routes(self) -> Set[str]:
        """Return known bus route IDs."""
        return self._bus_routes

    @property
    def bus_stops(self) -> List[str]:
        return self._bus_stops

    @property
    def train_stations(self) -> List[str]:
        return self._train_stations
