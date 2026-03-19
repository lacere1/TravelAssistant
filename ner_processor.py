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

from tfl_stop_datasets import TRAIN_LINES, load_bus_routes, load_bus_stops, load_train_stations

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
        # These domain lists are loaded from a dedicated dataset module so that
        # `NERProcessor` can be treated as a fallback component.
        self._bus_stops = load_bus_stops()
        self._train_stations = load_train_stations()
        self._bus_routes = load_bus_routes()
        self._train_lines = list(TRAIN_LINES)

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

        # Journey origin/destination etc. come from the LLM when available;
        # this path only runs when the LLM is unavailable (see nlp_processor.process).

        # ---- Phase 2: Domain-specific regex ----
        domain_entities = self._extract_domain_entities(text, text_lower)
        # Domain entities take precedence for domain-specific fields
        for key, value in domain_entities.items():
            if key not in entities or key in (
                "route", "line", "bus_route", "timetable_mode",
                "travel_mode", "road",
            ):
                entities[key] = value

        # ---- Reconcile SpaCy + regex locations ----
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
        # Journey origin/destination extraction is now handled exclusively
        # by the LLM-based extractor (llm_entity_extractor.py).
        # No rule-based extraction is performed here.

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
