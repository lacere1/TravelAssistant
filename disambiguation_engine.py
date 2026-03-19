"""
Unified Disambiguation Engine for London Transport Locations.

Provides a single, shared disambiguation pipeline for both the timetable path
(bus stops, train stations) and the journey planner path (origin/destination
location resolution). Replaces the three separate ad-hoc implementations that
previously lived in chatbot.py, transport_api.py, and journey_planner.py.

Architecture (two-stage ranking):
  Stage 1 — Geospatial filtering:
      If a spatial anchor is available (from Google Places or LLM-extracted
      "near_area" entity), candidates outside a configurable radius are dropped.
      The anchor represents where the user most likely meant geographically.

  Stage 2 — Multi-signal scoring:
      Remaining candidates are scored with a weighted combination of:
        (a) Exact / token-overlap string match on stop name or code  (weight 0.40)
        (b) Semantic similarity via sentence-transformers embedding   (weight 0.30)
        (c) SequenceMatcher fuzzy ratio                               (weight 0.15)
        (d) Geospatial proximity bonus (closer = higher)              (weight 0.15)

  Confidence thresholds decide outcome:
        >= 0.85  →  auto-resolve (silent pick, record in preferences)
        0.60–0.84 →  present top-N candidates for user to choose
        <  0.60  →  ask user to rephrase

Per-user preferences (frequent stops, last chosen) boost candidate scores
so the system learns from repeated interactions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Geospatial helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points."""
    rlat1, rlon1, rlat2, rlon2 = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DisambiguationCandidate:
    """A single candidate location returned by TfL or Google Places."""
    id: str
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    towards: Optional[str] = None
    direction: Optional[str] = None
    platform: Optional[str] = None
    modes: List[str] = field(default_factory=list)
    qualifier: Optional[str] = None
    label: Optional[str] = None
    # Score fields (populated during ranking)
    distance_km: Optional[float] = None
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict matching the existing option format in the codebase."""
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "label": self.label or self.name,
        }
        if self.lat is not None:
            d["lat"] = self.lat
        if self.lon is not None:
            d["lon"] = self.lon
        if self.towards:
            d["towards"] = self.towards
        if self.direction:
            d["direction"] = self.direction
        if self.platform:
            d["platform"] = self.platform
        if self.modes:
            d["modes"] = self.modes
        if self.qualifier:
            d["qualifier"] = self.qualifier
        if self.distance_km is not None:
            d["distance_km"] = round(self.distance_km, 3)
        d["score"] = round(self.score, 4)
        return d


@dataclass
class SpatialAnchor:
    """
    A geographic point representing where the user most likely meant.

    Can come from:
      - Google Places grounding of the user's query
      - LLM-extracted "near_area" entity grounded via Places
      - LLM-extracted "towards" entity grounded via Places
    """
    lat: float
    lng: float
    source: str = ""  # e.g. "places_query", "near_area", "towards"


@dataclass
class UserContext:
    """Per-user preferences that influence ranking."""
    last_chosen_stop_id: Optional[str] = None
    frequent_stops: Dict[str, int] = field(default_factory=dict)


@dataclass
class DisambiguationResult:
    """The output of the disambiguation engine."""
    resolved: bool  # True if a single candidate was auto-picked
    candidates: List[DisambiguationCandidate]  # ranked list
    top_score: float
    action: str  # "auto_resolved" | "present_options" | "ask_rephrase"
    chosen: Optional[DisambiguationCandidate] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolved": self.resolved,
            "action": self.action,
            "top_score": round(self.top_score, 4),
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

# Default weights for the scoring signals
_W_EXACT = 0.40
_W_SEMANTIC = 0.30
_W_FUZZY = 0.15
_W_GEO = 0.15

# Confidence thresholds
_THRESHOLD_AUTO = 0.85
_THRESHOLD_PRESENT = 0.60

# Geospatial filter radius (km)
_DEFAULT_RADIUS_KM = 1.5  # 1.5km for bus stops (area-level anchors like "near Kingsbury"
                           # can be 800m–1km from stops on the area's edge)
_WIDE_RADIUS_KM = 3.0     # 3km for train stations / journey planner locations

# Maximum candidates to present to user
_MAX_PRESENT = 5


class DisambiguationEngine:
    """
    Unified location disambiguation for London transport.

    Usage:
        engine = DisambiguationEngine(sentence_model=shared_model)
        result = engine.disambiguate(
            query="Lavender Avenue",
            candidates=[...],
            anchor=SpatialAnchor(lat=51.55, lng=-0.29, source="near_area"),
            user_context=UserContext(frequent_stops={"490001234A": 5}),
        )
        if result.resolved:
            # use result.chosen
        elif result.action == "present_options":
            # show result.candidates[:5] to user
        else:
            # ask user to rephrase
    """

    def __init__(
        self,
        sentence_model=None,
        auto_threshold: float = _THRESHOLD_AUTO,
        present_threshold: float = _THRESHOLD_PRESENT,
        default_radius_km: float = _DEFAULT_RADIUS_KM,
        wide_radius_km: float = _WIDE_RADIUS_KM,
        max_present: int = _MAX_PRESENT,
    ):
        """
        Args:
            sentence_model: A sentence-transformers SentenceTransformer instance
                            (shared with IntentClassifier for efficiency).
                            If None, semantic scoring is disabled and its weight
                            is redistributed to exact + fuzzy.
            auto_threshold:  Score above which we auto-resolve.
            present_threshold: Score above which we present options.
            default_radius_km: Geospatial filter radius for bus stops.
            wide_radius_km: Geospatial filter radius for stations/journey locations.
            max_present: Max candidates to present to the user.
        """
        self._sentence_model = sentence_model
        self._auto_threshold = auto_threshold
        self._present_threshold = present_threshold
        self._default_radius_km = default_radius_km
        self._wide_radius_km = wide_radius_km
        self._max_present = max_present

        # Adjust weights if no sentence model
        if self._sentence_model is None:
            self._w_exact = _W_EXACT + _W_SEMANTIC * 0.6
            self._w_semantic = 0.0
            self._w_fuzzy = _W_FUZZY + _W_SEMANTIC * 0.4
            self._w_geo = _W_GEO
        else:
            self._w_exact = _W_EXACT
            self._w_semantic = _W_SEMANTIC
            self._w_fuzzy = _W_FUZZY
            self._w_geo = _W_GEO

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def disambiguate(
        self,
        query: str,
        candidates: List[DisambiguationCandidate],
        anchor: Optional[SpatialAnchor] = None,
        user_context: Optional[UserContext] = None,
        mode: str = "bus",
        radius_km: Optional[float] = None,
    ) -> DisambiguationResult:
        """
        Run the full disambiguation pipeline.

        Args:
            query: The user's original location query string.
            candidates: List of candidate locations from TfL / Places API.
            anchor: Optional spatial anchor for geospatial filtering.
            user_context: Optional per-user preferences.
            mode: "bus", "train", or "journey" — affects default radius.
            radius_km: Override geospatial filter radius (km).

        Returns:
            DisambiguationResult with ranked candidates and action.
        """
        if not candidates:
            return DisambiguationResult(
                resolved=False, candidates=[], top_score=0.0, action="ask_rephrase"
            )

        if len(candidates) == 1:
            candidates[0].score = 1.0
            return DisambiguationResult(
                resolved=True,
                candidates=candidates,
                top_score=1.0,
                action="auto_resolved",
                chosen=candidates[0],
            )

        # Determine radius
        if radius_km is None:
            radius_km = self._wide_radius_km if mode in ("train", "journey") else self._default_radius_km

        # Stage 1: Geospatial filtering
        filtered = self._geospatial_filter(candidates, anchor, radius_km)

        # If filtering removed everything, fall back to all candidates
        # (anchor might be wrong or too restrictive)
        if not filtered:
            filtered = candidates

        # Stage 2: Multi-signal scoring
        scored = self._score_candidates(query, filtered, anchor, user_context)

        # Sort by score descending
        scored.sort(key=lambda c: c.score, reverse=True)

        top_score = scored[0].score if scored else 0.0

        # Decision logic
        if top_score >= self._auto_threshold:
            # Check the gap between #1 and #2 — only auto-resolve if clear winner
            if len(scored) >= 2:
                gap = scored[0].score - scored[1].score
                if gap < 0.10:
                    # Too close — present options instead
                    return DisambiguationResult(
                        resolved=False,
                        candidates=scored[: self._max_present],
                        top_score=top_score,
                        action="present_options",
                    )
            return DisambiguationResult(
                resolved=True,
                candidates=scored[: self._max_present],
                top_score=top_score,
                action="auto_resolved",
                chosen=scored[0],
            )
        elif top_score >= self._present_threshold:
            return DisambiguationResult(
                resolved=False,
                candidates=scored[: self._max_present],
                top_score=top_score,
                action="present_options",
            )
        else:
            return DisambiguationResult(
                resolved=False,
                candidates=scored[: self._max_present],
                top_score=top_score,
                action="ask_rephrase",
            )

    # ------------------------------------------------------------------
    # Stage 1: Geospatial filtering
    # ------------------------------------------------------------------

    def _geospatial_filter(
        self,
        candidates: List[DisambiguationCandidate],
        anchor: Optional[SpatialAnchor],
        radius_km: float,
    ) -> List[DisambiguationCandidate]:
        """
        Remove candidates that are geographically too far from the anchor.
        Candidates without coordinates are always kept.
        """
        if anchor is None:
            return list(candidates)

        kept: List[DisambiguationCandidate] = []
        for c in candidates:
            if c.lat is None or c.lon is None:
                # No coordinates — keep it (can't filter)
                kept.append(c)
                continue
            dist = haversine_km(anchor.lat, anchor.lng, c.lat, c.lon)
            c.distance_km = dist
            if dist <= radius_km:
                kept.append(c)

        return kept

    # ------------------------------------------------------------------
    # Stage 2: Multi-signal scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        query: str,
        candidates: List[DisambiguationCandidate],
        anchor: Optional[SpatialAnchor],
        user_context: Optional[UserContext],
    ) -> List[DisambiguationCandidate]:
        """Score each candidate with the weighted combination of signals."""
        query_lower = query.strip().lower()
        query_tokens = set(self._tokenize(query_lower))

        # Pre-compute semantic embeddings if model available
        semantic_scores = self._compute_semantic_scores(query, candidates)

        # Find max distance for normalization (among candidates that have coordinates)
        max_dist = 0.0
        for c in candidates:
            if c.distance_km is not None and c.distance_km > max_dist:
                max_dist = c.distance_km

        for i, c in enumerate(candidates):
            # (a) Exact / token-overlap string match
            exact_score = self._exact_match_score(query_lower, query_tokens, c)

            # (b) Semantic similarity
            sem_score = semantic_scores[i] if semantic_scores else 0.0

            # (c) Fuzzy ratio
            fuzzy_score = self._fuzzy_score(query_lower, c)

            # (d) Geospatial proximity (closer = higher)
            geo_score = self._geo_proximity_score(c, max_dist)

            # Weighted combination
            raw_score = (
                self._w_exact * exact_score
                + self._w_semantic * sem_score
                + self._w_fuzzy * fuzzy_score
                + self._w_geo * geo_score
            )

            # User preference boost (additive, capped)
            pref_boost = self._preference_boost(c, user_context)
            c.score = min(1.0, raw_score + pref_boost)

        return candidates

    # ------------------------------------------------------------------
    # Scoring signal: exact / token-overlap
    # ------------------------------------------------------------------

    def _exact_match_score(
        self,
        query_lower: str,
        query_tokens: set,
        candidate: DisambiguationCandidate,
    ) -> float:
        """
        Score based on exact string and token-level matching.

        Returns 0.0–1.0 where:
          1.0 = exact match on name or stop code
          0.8 = all query tokens appear in candidate name
          0.5–0.8 = partial token overlap
          0.0 = no overlap
        """
        name_lower = (candidate.name or "").lower()
        label_lower = (candidate.label or candidate.name or "").lower()

        # Perfect exact match
        if query_lower == name_lower or query_lower == label_lower:
            return 1.0

        # Query is a substring of name or vice versa
        if query_lower in name_lower or name_lower in query_lower:
            return 0.9

        # Token overlap
        name_tokens = set(self._tokenize(name_lower))
        if not name_tokens or not query_tokens:
            return 0.0

        overlap = query_tokens & name_tokens
        if overlap == query_tokens:
            # All query tokens appear in name
            return 0.8

        # Partial overlap: Jaccard-like
        union = query_tokens | name_tokens
        jaccard = len(overlap) / len(union) if union else 0.0
        return jaccard * 0.7

    # ------------------------------------------------------------------
    # Scoring signal: semantic similarity
    # ------------------------------------------------------------------

    def _compute_semantic_scores(
        self,
        query: str,
        candidates: List[DisambiguationCandidate],
    ) -> Optional[List[float]]:
        """Compute cosine similarity between query and each candidate name."""
        if self._sentence_model is None or not candidates:
            return None

        try:
            texts = [query] + [c.label or c.name or "" for c in candidates]
            embeddings = self._sentence_model.encode(texts, show_progress_bar=False)

            query_emb = embeddings[0]
            scores: List[float] = []
            for i in range(1, len(embeddings)):
                sim = self._cosine_similarity(query_emb, embeddings[i])
                # Normalize to 0–1 range (cosine sim for text is typically 0–1)
                scores.append(max(0.0, min(1.0, sim)))
            return scores
        except Exception as e:
            print(f"[DisambiguationEngine] Semantic scoring failed: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Scoring signal: fuzzy string matching
    # ------------------------------------------------------------------

    def _fuzzy_score(
        self,
        query_lower: str,
        candidate: DisambiguationCandidate,
    ) -> float:
        """SequenceMatcher ratio between query and candidate name/label."""
        name = (candidate.name or "").lower()
        label = (candidate.label or "").lower()

        score_name = SequenceMatcher(None, query_lower, name).ratio()
        score_label = SequenceMatcher(None, query_lower, label).ratio() if label != name else 0.0

        # Also check towards field (for "bus times towards X" queries)
        towards = (candidate.towards or "").lower()
        score_towards = SequenceMatcher(None, query_lower, towards).ratio() if towards else 0.0

        return max(score_name, score_label, score_towards)

    # ------------------------------------------------------------------
    # Scoring signal: geospatial proximity
    # ------------------------------------------------------------------

    def _geo_proximity_score(
        self,
        candidate: DisambiguationCandidate,
        max_distance: float,
    ) -> float:
        """
        Convert distance to a 0–1 score where closer = higher.
        Candidates without distance info get a neutral 0.5.
        """
        if candidate.distance_km is None:
            return 0.5

        if max_distance <= 0:
            return 1.0

        # Inverse linear: 0 km = 1.0, max_distance km = 0.0
        return max(0.0, 1.0 - (candidate.distance_km / max(max_distance, 0.001)))

    # ------------------------------------------------------------------
    # User preference boost
    # ------------------------------------------------------------------

    def _preference_boost(
        self,
        candidate: DisambiguationCandidate,
        user_context: Optional[UserContext],
    ) -> float:
        """
        Small additive boost for stops the user has chosen before.
        Max boost: 0.10 (so it nudges but doesn't override signals).
        """
        if user_context is None:
            return 0.0

        boost = 0.0

        # Last chosen stop gets a boost
        if user_context.last_chosen_stop_id and candidate.id == user_context.last_chosen_stop_id:
            boost += 0.05

        # Frequent stop gets a boost proportional to count (capped)
        count = user_context.frequent_stops.get(candidate.id, 0)
        if count > 0:
            # Logarithmic scaling: boost grows slowly with frequency
            boost += min(0.05, 0.02 * math.log1p(count))

        return min(boost, 0.10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _STOP_WORDS = frozenset({
        "stop", "station", "underground", "bus", "rail", "dlr",
        "overground", "tram", "the", "a", "an", "at", "in", "on",
        "to", "from", "near", "towards", "for", "of",
    })

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Tokenize and remove stop words."""
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [w for w in words if w not in cls._STOP_WORDS and len(w) >= 2]


# ---------------------------------------------------------------------------
# Helper: build candidates from TfL StopPoint search matches
# ---------------------------------------------------------------------------

def candidates_from_tfl_matches(matches: List[Dict[str, Any]]) -> List[DisambiguationCandidate]:
    """Convert TfL StopPoint search results to DisambiguationCandidate list."""
    result = []
    for m in matches:
        result.append(DisambiguationCandidate(
            id=m.get("id", ""),
            name=m.get("name", "Unknown"),
            lat=m.get("lat"),
            lon=m.get("lon"),
            towards=m.get("towards"),
            direction=m.get("direction"),
            platform=m.get("platform"),
            modes=m.get("modes", []),
            label=m.get("label") or m.get("name", "Unknown"),
        ))
    return result


def candidates_from_journey_options(options: List[Dict[str, Any]]) -> List[DisambiguationCandidate]:
    """Convert journey planner disambiguation options to DisambiguationCandidate list."""
    result = []
    for o in options:
        lat, lng = None, None
        # Some options encode coordinates in the id as "lat,lng"
        oid = o.get("id", "")
        if "," in oid:
            parts = oid.split(",")
            if len(parts) == 2:
                try:
                    lat, lng = float(parts[0]), float(parts[1])
                except ValueError:
                    pass
        # Prefer explicit lat/lng if available
        if o.get("lat") is not None:
            lat = o["lat"]
        if o.get("lng") is not None:
            lng = o["lng"]
        elif o.get("lon") is not None:
            lng = o["lon"]

        result.append(DisambiguationCandidate(
            id=oid,
            name=o.get("name", "Unknown"),
            lat=lat,
            lon=lng,
            qualifier=o.get("qualifier"),
            label=o.get("shortLabel") or o.get("label") or o.get("name", "Unknown"),
        ))
    return result


# ---------------------------------------------------------------------------
# Helper: build SpatialAnchor from Places grounding or LLM entities
# ---------------------------------------------------------------------------

def anchor_from_places(places_result: Optional[Dict[str, Any]], source: str = "places_query") -> Optional[SpatialAnchor]:
    """Create a SpatialAnchor from a PlacesGrounder result dict."""
    if not places_result:
        return None
    lat = places_result.get("lat")
    lng = places_result.get("lng")
    if lat is not None and lng is not None:
        return SpatialAnchor(lat=lat, lng=lng, source=source)
    return None


def anchor_from_coordinates(lat: Optional[float], lng: Optional[float], source: str = "coordinates") -> Optional[SpatialAnchor]:
    """Create a SpatialAnchor from explicit coordinates."""
    if lat is not None and lng is not None:
        return SpatialAnchor(lat=lat, lng=lng, source=source)
    return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[DisambiguationEngine] = None


def get_disambiguation_engine(sentence_model=None) -> DisambiguationEngine:
    """Get or create the singleton DisambiguationEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DisambiguationEngine(sentence_model=sentence_model)
    return _engine_instance
