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
    place_types: List[str] = field(default_factory=list)
    qualifier: Optional[str] = None
    label: Optional[str] = None
    # Score fields (populated during ranking)
    distance_km: Optional[float] = None
    score: float = 0.0
    # Fuzzy match of query vs candidate name (SequenceMatcher ratio); set in journey scoring
    name_similarity: Optional[float] = None

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
        if self.place_types:
            d["place_types"] = self.place_types
        if self.qualifier:
            d["qualifier"] = self.qualifier
        if self.distance_km is not None:
            d["distance_km"] = round(self.distance_km, 3)
        d["score"] = round(self.score, 4)
        if self.name_similarity is not None:
            d["name_similarity"] = round(self.name_similarity, 4)
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

# --- Non-normalised additive scoring model ---
#
# Weights represent each signal's *maximum contribution* to the final score.
# Absent signals are NOT redistributed to the remaining ones — they contribute
# a small neutral constant (_NEUTRAL_GEO) instead.  This means scores honestly
# reflect how much information is available:
#
#   • Perfect name match, no geo at all → ~0.72  ("name matches but I don't
#     know where you are")
#   • Perfect name match + user nearby  → ~0.90+  (clearly the right area)
#   • Perfect name match + anchor + user nearby → caps at 1.0 (very confident)
#
# This prevents same-name candidates all scoring an identical 90% and makes
# the user-proximity signal a real differentiator.
_W_EXACT      = 0.25   # name: exact / token-overlap
_W_FUZZY      = 0.08   # name: SequenceMatcher ratio
_W_GEO_ANCHOR = 0.28   # anchor proximity (near_area grounding)
_W_USER_PROX  = 0.38   # user device proximity
# When towards is specified, exact/fuzzy shrink to make room without
# the total exceeding 1.0 (score is still capped at 1.0 anyway).
_W_TOWARDS    = 0.28
_TOWARDS_NAME_SHRINK = 0.70   # multiply exact & fuzzy by this when towards active

# Neutral contribution for absent geo signals (rather than 0 or redistribution)
_NEUTRAL_GEO = 0.36   # "unknown distance" — keeps name-led scores usable

# Confidence thresholds (tuned for the non-normalised model)
_THRESHOLD_AUTO    = 0.84   # slightly above old 0.82 — name weights are higher
_THRESHOLD_PRESENT = 0.30   # show options if any reasonable match

# Geospatial filter radius (km)
_DEFAULT_RADIUS_KM = 1.5  # 1.5km for bus stops (area-level anchors like "near Kingsbury"
                           # can be 800m–1km from stops on the area's edge)
_WIDE_RADIUS_KM = 3.0     # 3km for train stations / journey planner locations

# Maximum candidates to present to user
_MAX_PRESENT = 5
_DEDUP_SCORE_EPSILON = 0.02


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
            sentence_model: Unused (kept for backward-compat). Semantic scoring
                            has been removed; exact + fuzzy + geo + towards are
                            the active signals.
            auto_threshold:  Score above which we auto-resolve.
            present_threshold: Score above which we present options.
            default_radius_km: Geospatial filter radius for bus stops.
            wide_radius_km: Geospatial filter radius for stations/journey locations.
            max_present: Max candidates to present to the user.
        """
        self._sentence_model = sentence_model  # retained for API compat
        self._auto_threshold = auto_threshold
        self._present_threshold = present_threshold
        self._default_radius_km = default_radius_km
        self._wide_radius_km = wide_radius_km
        self._max_present = max_present
        # Weights are module-level constants; no instance copies needed.

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
        towards_query: Optional[str] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
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
            towards_query: Optional direction text (e.g. "Wembley"). When
                           present, a towards-match signal is injected.
            user_lat: Device latitude (from browser geolocation).
            user_lon: Device longitude (from browser geolocation).

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

        # Journey mode (temporary): disable ranking/dedup heuristics and
        # simply present the first 10 candidates after geospatial filtering.
        if mode == "journey":
            unranked = filtered[:10]
            if not unranked:
                return DisambiguationResult(
                    resolved=False, candidates=[], top_score=0.0, action="ask_rephrase"
                )
            for c in unranked:
                c.score = 0.0
            return DisambiguationResult(
                resolved=False,
                candidates=unranked,
                top_score=0.0,
                action="present_options",
            )

        # Non-journey modes (bus/train timetable): full scoring pipeline.

        # Stage 2: Multi-signal scoring
        scored = self._score_candidates(
            query, filtered, anchor, user_context, towards_query, user_lat, user_lon
        )

        # Sort by score descending
        scored.sort(key=lambda c: c.score, reverse=True)

        top_score = scored[0].score if scored else 0.0

        # Decision logic
        if top_score >= self._auto_threshold:
            # Check the gap between #1 and #2 — only auto-resolve if clear winner
            if len(scored) >= 2:
                gap = scored[0].score - scored[1].score
                if gap < 0.10:
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
        towards_query: Optional[str] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
    ) -> List[DisambiguationCandidate]:
        """Score each candidate using a non-normalised additive model.

        Non-normalised means absent signals contribute _NEUTRAL_GEO (0.36)
        rather than zero or redistributed weight. This keeps scores honest:
          • Perfect name match, no geo  → ~0.72  (name matches, location unknown)
          • Perfect name match + anchor → ~0.90  (we know the area)
          • Perfect name match + user nearby → ~0.90  (device is close)
          • Perfect name + anchor + user nearby → ~1.0  (capped; very confident)

        When towards_query is provided, exact/fuzzy are shrunk by
        _TOWARDS_NAME_SHRINK and a towards signal is added at _W_TOWARDS.
        """
        query_lower = query.strip().lower()
        query_tokens = set(self._tokenize(query_lower))

        has_towards = bool(towards_query and towards_query.strip())
        has_anchor = anchor is not None
        has_user_geo = (user_lat is not None and user_lon is not None)

        if has_towards:
            w_exact = _W_EXACT * _TOWARDS_NAME_SHRINK
            w_fuzzy = _W_FUZZY * _TOWARDS_NAME_SHRINK
            w_towards = _W_TOWARDS
            towards_lower = towards_query.strip().lower()
            towards_tokens = set(self._tokenize(towards_lower))
            print(
                f"[DisambiguationEngine] Scoring with towards='{towards_query}' — "
                f"weights: exact={w_exact:.2f} fuzzy={w_fuzzy:.2f} "
                f"anchor={_W_GEO_ANCHOR:.2f} user_prox={_W_USER_PROX:.2f} "
                f"towards={w_towards:.2f}"
            )
        else:
            w_exact = _W_EXACT
            w_fuzzy = _W_FUZZY
            w_towards = 0.0
            towards_lower = ""
            towards_tokens = set()

        for c in candidates:
            # (a) Exact / token-overlap name match
            exact_score = self._exact_match_score(query_lower, query_tokens, c)

            # (b) Fuzzy SequenceMatcher ratio
            fuzzy_score = self._fuzzy_score(query_lower, c)

            # (c) Anchor proximity — absolute stepped decay
            if has_anchor and c.lat is not None and c.lon is not None:
                # distance_km was set during geospatial filter; recompute if missing
                if c.distance_km is None:
                    c.distance_km = haversine_km(anchor.lat, anchor.lng, c.lat, c.lon)
                anchor_score = self._stepped_geo_score(c.distance_km)
            else:
                anchor_score = _NEUTRAL_GEO  # unknown — neutral, not punishing

            # (d) User device proximity — absolute stepped decay
            if has_user_geo and c.lat is not None and c.lon is not None:
                user_dist_km = haversine_km(user_lat, user_lon, c.lat, c.lon)
                user_prox_score = self._stepped_geo_score(user_dist_km)
            else:
                user_prox_score = _NEUTRAL_GEO  # device location unknown — neutral

            # (e) Towards / direction match
            towards_score = 0.0
            if has_towards:
                towards_score = self._towards_match_score(
                    towards_lower, towards_tokens, c
                )

            # Non-normalised additive combination (cap at 1.0)
            raw_score = (
                w_exact * exact_score
                + w_fuzzy * fuzzy_score
                + _W_GEO_ANCHOR * anchor_score
                + _W_USER_PROX * user_prox_score
                + w_towards * towards_score
            )

            pref_boost = self._preference_boost(c, user_context)
            c.score = min(1.0, raw_score + pref_boost)

            print(
                f"[DisambiguationEngine] '{c.name}' "
                f"exact={exact_score:.2f} fuzzy={fuzzy_score:.2f} "
                f"anchor={anchor_score:.2f} user_prox={user_prox_score:.2f}"
                + (f" towards={towards_score:.2f}" if has_towards else "")
                + f" pref={pref_boost:.3f} → score={c.score:.4f}"
            )

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
    # Scoring signal: towards / direction match
    # ------------------------------------------------------------------

    def _towards_match_score(
        self,
        towards_lower: str,
        towards_tokens: set,
        candidate: DisambiguationCandidate,
    ) -> float:
        """Score how well a candidate's direction matches the user's towards query.

        Uses the same token-matching approach as the journey planner's context
        signal — the candidate's ``towards`` field (set from TfL search data or
        live arrivals) is compared against the user's direction text.

        Returns 0.0–1.0 where:
          1.0 = exact match or full substring containment
          0.8 = all towards-query tokens appear in candidate towards
          0.0–0.8 = proportional token overlap
          0.0 = no overlap or candidate has no towards field
        """
        cand_towards = (candidate.towards or "").strip().lower()
        if not cand_towards or not towards_lower:
            return 0.0

        # Perfect match
        if towards_lower == cand_towards or towards_lower in cand_towards:
            return 1.0
        if cand_towards in towards_lower:
            return 0.9

        # Token overlap — using the noise-stripped tokenizer
        cand_tokens = set(self._tokenize(cand_towards))
        if not towards_tokens or not cand_tokens:
            return 0.0

        overlap = towards_tokens & cand_tokens
        if overlap == towards_tokens:
            return 0.8

        # Proportional overlap
        return len(overlap) / len(towards_tokens) * 0.7

    # ------------------------------------------------------------------
    # Scoring signal: geospatial proximity
    # ------------------------------------------------------------------

    @staticmethod
    def _stepped_geo_score(dist_km: Optional[float]) -> float:
        """
        Convert a distance in km to a 0–1 score using stepped decay.

        Farther locations score strictly lower so anchor / device proximity can
        separate same-name stops. Steps are tighter than a flat borough-wide band.

          ≤ 0.25 km →  1.00
          ≤ 0.50 km →  0.88
          ≤ 1.00 km →  0.70
          ≤ 2.00 km →  0.48
          ≤ 3.50 km →  0.28
          ≤ 5.00 km →  0.12
          ≤ 8.00 km →  0.03
            > 8.0 km →  0.00
        """
        if dist_km is None:
            return _NEUTRAL_GEO
        if dist_km <= 0.25:
            return 1.0
        if dist_km <= 0.5:
            return 0.88
        if dist_km <= 1.0:
            return 0.70
        if dist_km <= 2.0:
            return 0.48
        if dist_km <= 3.5:
            return 0.28
        if dist_km <= 5.0:
            return 0.12
        if dist_km <= 8.0:
            return 0.03
        return 0.0

    def _geo_proximity_score(
        self,
        candidate: DisambiguationCandidate,
        max_distance: float,
    ) -> float:
        """Legacy shim — delegates to _stepped_geo_score. Kept for API compat."""
        return self._stepped_geo_score(candidate.distance_km)

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
    # Journey-mode scoring
    # ------------------------------------------------------------------

    def _journey_score_candidates(
        self,
        query: str,
        candidates: List[DisambiguationCandidate],
    ) -> List[DisambiguationCandidate]:
        """Score journey candidates by name similarity to the user query.

        Signals:
        1. Name similarity (weight 0.80) — SequenceMatcher ratio between
           the normalised query and candidate name.
        2. Address bonus  (weight 0.20) — proportion of query tokens found
           in the candidate's formattedAddress that are *not* already in
           the candidate name.  Rewards candidates whose address matches
           area hints the user typed (e.g. "Mitcham" in the query matching
           the address "Mitcham, London CR4").
        """
        q_norm = self._normalize_place_name(query)
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

        for c in candidates:
            c_norm = self._normalize_place_name(c.name)
            name_sim = SequenceMatcher(None, q_norm, c_norm).ratio()

            addr_bonus = 0.0
            addr = (c.qualifier or "").lower()
            if addr and q_tokens:
                name_tokens = set(re.findall(r"[a-z0-9]+", (c.name or "").lower()))
                extra_query_tokens = q_tokens - name_tokens
                if extra_query_tokens:
                    addr_tokens = set(re.findall(r"[a-z0-9]+", addr))
                    matched = extra_query_tokens & addr_tokens
                    addr_bonus = len(matched) / len(extra_query_tokens)

            combined = (0.80 * name_sim) + (0.20 * addr_bonus)
            c.score = round(combined, 4)
            print(
                f"[JourneyScore] '{c.name}' | name_sim={name_sim:.4f} "
                f"addr_bonus={addr_bonus:.4f} | score={c.score}"
            )

        return candidates

    # ------------------------------------------------------------------
    # Geographic spread detection (journey mode)
    # ------------------------------------------------------------------

    # Spread threshold must sit just above the dedup proximity threshold (0.15 km)
    # so that any two similar-name candidates that were NOT merged by dedup
    # (because they are more than 150 m apart) are flagged here and presented
    # to the user rather than silently auto-resolved.
    _SPREAD_THRESHOLD_KM = 0.20

    def _has_geographic_spread(
        self,
        candidates: List[DisambiguationCandidate],
    ) -> bool:
        """
        Return True if candidates contain similar-name locations that are
        geographically far apart (> _SPREAD_THRESHOLD_KM).

        Threshold is set just above the dedup proximity threshold (0.15 km) so
        the two checks are contiguous:
          <= 0.15 km  →  deduplicated (same physical spot)
          >  0.20 km  →  spread detected → present options to user

        This closes the previous gap (0.15–1.0 km) where candidates were
        neither merged nor flagged, causing silent wrong auto-resolves.
        """
        coords = [
            (c, c.lat, c.lon)
            for c in candidates
            if c.lat is not None and c.lon is not None
        ]
        if len(coords) < 2:
            return False

        for i in range(len(coords)):
            c_i, lat_i, lon_i = coords[i]
            name_i = self._normalize_place_name(c_i.name)
            for j in range(i + 1, len(coords)):
                c_j, lat_j, lon_j = coords[j]
                name_j = self._normalize_place_name(c_j.name)

                name_sim = SequenceMatcher(None, name_i, name_j).ratio()
                if name_sim < 0.70:
                    continue

                dist = haversine_km(lat_i, lon_i, lat_j, lon_j)
                if dist > self._SPREAD_THRESHOLD_KM:
                    print(
                        f"[DisambiguationEngine] Spread: '{c_i.name}' ↔ "
                        f"'{c_j.name}' = {dist:.1f} km apart (name_sim={name_sim:.2f})"
                    )
                    return True

        return False

    # ------------------------------------------------------------------
    # Stage 1.5: Candidate deduplication (journey mode)
    # ------------------------------------------------------------------

    _TRANSPORT_SUFFIXES = [
        "underground station", "rail station", "bus station",
        "dlr station", "tube station", "overground station",
        "tram stop", "bus stop", "train station",
        "station", "stop",
    ]
    _BUS_STOP_TYPES = frozenset({"bus_stop"})
    _TRANSIT_TYPES = frozenset({
        "bus_stop",
        "bus_station",
        "train_station",
        "subway_station",
        "light_rail_station",
        "transit_station",
        "tram_stop",
    })

    @classmethod
    def _normalize_place_name(cls, name: str) -> str:
        """Strip transport suffixes and normalise for dedup comparison."""
        n = (name or "").lower().strip()
        n = re.sub(r",?\s*london\s*$", "", n).strip()
        for suffix in cls._TRANSPORT_SUFFIXES:
            if n.endswith(suffix):
                n = n[: -len(suffix)].strip().rstrip(",").strip()
                break
        return n

    @classmethod
    def _is_bus_stop_candidate(cls, c: DisambiguationCandidate) -> bool:
        place_types = set((c.place_types or []))
        if place_types & cls._BUS_STOP_TYPES:
            return True
        return any((m or "").lower() == "bus" for m in (c.modes or []))

    @classmethod
    def _is_transit_candidate(cls, c: DisambiguationCandidate) -> bool:
        place_types = set((c.place_types or []))
        if place_types & cls._TRANSIT_TYPES:
            return True
        return any(
            (m or "").lower() in {"bus", "tube", "overground", "dlr", "tram", "train", "rail"}
            for m in (c.modes or [])
        )

    @classmethod
    def _query_prefers_bus_stop(cls, query: str) -> Optional[bool]:
        """
        Infer whether user wording prefers bus-stop style results.
        Returns True (bus), False (non-transit), or None (no clear preference).
        """
        q = (query or "").lower()
        bus_signals = ("bus stop", "bus station", "stop", "stand ", "bay ", "route ", "bus ")
        non_transit_signals = ("road", "street", "avenue", "lane", "close", "drive", "court", "crescent", "way", "place")
        if any(s in q for s in bus_signals):
            return True
        if any(s in q for s in non_transit_signals):
            return False
        return None

    def _preference_boost_for_dedup_choice(self, query: str, c: DisambiguationCandidate) -> float:
        pref = self._query_prefers_bus_stop(query)
        if pref is True:
            return 0.03 if self._is_bus_stop_candidate(c) else 0.0
        if pref is False:
            return 0.03 if not self._is_transit_candidate(c) else 0.0
        return 0.0

    def _deduplicate_candidates(
        self,
        query: str,
        candidates: List[DisambiguationCandidate],
    ) -> List[DisambiguationCandidate]:
        """Merge candidates that represent the same physical location.

        Two candidates are duplicates when:
          - Both have coordinates within 150 m of each other, AND
          - Their names are substantially similar after stripping transport
            suffixes (SequenceMatcher ratio > 0.75).

        Among duplicates the candidate with the highest pre-computed
        ``score`` (set by ``_journey_score_candidates``) survives.
        """
        if len(candidates) <= 1:
            return candidates

        kept: List[DisambiguationCandidate] = []

        for c in candidates:
            if c.lat is None or c.lon is None:
                kept.append(c)
                continue

            c_norm = self._normalize_place_name(c.name)
            merged = False

            for i, k in enumerate(kept):
                if k.lat is None or k.lon is None:
                    continue

                dist = haversine_km(c.lat, c.lon, k.lat, k.lon)
                if dist > 0.15:
                    continue

                k_norm = self._normalize_place_name(k.name)
                sim = SequenceMatcher(None, c_norm, k_norm).ratio()
                if sim < 0.75:
                    continue

                score_c = (c.score or 0)
                score_k = (k.score or 0)
                score_gap = score_c - score_k

                # Near-ties can occur between a bus stop and a non-transit place
                # representing almost the same location. Use query intent to break
                # ties consistently instead of relying on candidate arrival order.
                if abs(score_gap) <= _DEDUP_SCORE_EPSILON:
                    pref_c = self._preference_boost_for_dedup_choice(query, c)
                    pref_k = self._preference_boost_for_dedup_choice(query, k)
                    if pref_c > pref_k:
                        score_gap = 1.0
                    elif pref_k > pref_c:
                        score_gap = -1.0
                    else:
                        score_gap = 1.0 if (c.name or "").lower() > (k.name or "").lower() else -1.0

                if score_gap > 0:
                    print(
                        f"[Dedup] Replacing '{k.name}' (score={k.score}) "
                        f"with '{c.name}' (score={c.score})"
                    )
                    kept[i] = c
                else:
                    print(
                        f"[Dedup] Keeping '{k.name}' (score={k.score}), "
                        f"dropping '{c.name}' (score={c.score})"
                    )
                merged = True
                break

            if not merged:
                kept.append(c)

        if len(kept) < len(candidates):
            print(
                f"[DisambiguationEngine] Dedup: {len(candidates)} → {len(kept)} candidates "
                f"(merged {len(candidates) - len(kept)} near-duplicates)"
            )

        return kept

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

        ns = o.get("name_similarity")
        try:
            name_sim = float(ns) if ns is not None else None
        except (TypeError, ValueError):
            name_sim = None
        result.append(DisambiguationCandidate(
            id=oid,
            name=o.get("name", "Unknown"),
            lat=lat,
            lon=lng,
            qualifier=o.get("qualifier"),
            label=o.get("shortLabel") or o.get("label") or o.get("name", "Unknown"),
            place_types=o.get("place_types") or [],
            name_similarity=name_sim,
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
# Journey Disambiguation: weighted scoring, road grouping, outcomes
# ---------------------------------------------------------------------------

@dataclass
class JourneyUserHistory:
    """Per-user journey history for scoring boosts.

    The frontend persists this in localStorage and sends it with each
    request.  The backend receives it as a plain dict and converts here.
    """
    chosen_locations: Dict[str, int] = field(default_factory=dict)
    # Maps location id (e.g. "51.585,-0.278") → selection count

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "JourneyUserHistory":
        if not d:
            return cls()
        return cls(chosen_locations=d.get("chosen_locations", {}))


@dataclass
class NearAreaStructured:
    """Structured address fields extracted from a grounded near_area result.

    These come from Nominatim's addressdetails and allow hard-boundary
    matching against candidate addresses — far more precise than distance
    decay from an arbitrary centroid.

    Example: grounding "Sudbury" yields:
        suburb="Sudbury", borough="London Borough of Brent", postcode_prefix="HA0"

    A candidate whose address contains "HA0" is almost certainly in Sudbury;
    one containing "London Borough of Brent" is at least in the right borough.
    """
    suburb: str = ""           # neighbourhood name  e.g. "Sudbury", "Kingsbury"
    borough: str = ""          # borough name        e.g. "London Borough of Brent"
    postcode_prefix: str = ""  # outward code        e.g. "HA0", "NW9", "W2"


@dataclass
class JourneyDisambiguationContext:
    """All the contextual signals available when disambiguating a journey
    location (origin or destination)."""
    role: str                                   # "origin" or "destination"
    user_lat: Optional[float] = None            # browser geolocation
    user_lon: Optional[float] = None
    other_end_lat: Optional[float] = None       # the resolved other endpoint
    other_end_lon: Optional[float] = None
    near_area_anchor: Optional[SpatialAnchor] = None  # from "near X" context — lat/lng
    near_area_text: Optional[str] = None              # raw text e.g. "Sudbury" — for loose address matching
    near_area_structured: Optional[NearAreaStructured] = None  # grounded structured fields — hard boundary matching
    user_history: Optional[JourneyUserHistory] = None


@dataclass
class RoadGroup:
    """A cluster of nearby candidates on the same road / in the same area."""
    road_name: str
    area: str
    candidates: List[DisambiguationCandidate]
    # Bounding box for the group (for polygon display)
    min_lat: float = 0.0
    max_lat: float = 0.0
    min_lon: float = 0.0
    max_lon: float = 0.0
    representative_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road_name": self.road_name,
            "area": self.area,
            "candidates": [c.to_dict() for c in self.candidates],
            "bounds": {
                "min_lat": round(self.min_lat, 6),
                "max_lat": round(self.max_lat, 6),
                "min_lon": round(self.min_lon, 6),
                "max_lon": round(self.max_lon, 6),
            },
            "representative_score": round(self.representative_score, 4),
        }


@dataclass
class JourneyDisambiguationResult:
    """Result of the journey-specific disambiguation pipeline."""
    action: str               # "auto_select" | "clarify" | "show_list"
    candidates: List[DisambiguationCandidate]    # ranked
    top_score: float
    chosen: Optional[DisambiguationCandidate] = None
    confidence_gap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "top_score": round(self.top_score, 4),
            "confidence_gap": round(self.confidence_gap, 4),
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# Default weights for journey disambiguation scoring.
# These are the "ideal" weights when ALL signals are available.
# When a signal is unavailable, its weight is redistributed proportionally
# to the remaining signals so the total always sums to 1.0 and missing
# data never drags down scores.
_JW_PROXIMITY_USER = 0.20       # proximity to user's current location
_JW_PROXIMITY_OTHER = 0.15      # proximity to the other end (dest if origin, origin if dest)
_JW_CONTEXT_MATCH = 0.30        # context clues ("near kingsbury") — user explicitly specified area
_JW_NAME_MATCH = 0.20           # name similarity to query
_JW_USER_HISTORY = 0.15         # user history boost
_JW_ROUTE_FEASIBILITY = 0.05    # route feasibility (reserved for future TfL integration)

# Thresholds for journey disambiguation outcomes
_J_AUTO_SELECT_THRESHOLD = 0.82
_J_AUTO_SELECT_GAP = 0.20       # minimum gap between #1 and #2 for auto-select
_J_CLARIFY_GAP = 0.15           # if gap < this, ask for clarification
_J_SHOW_LIST_THRESHOLD = 0.30   # below this, show full list

# Road grouping constants
_ROAD_GROUP_DISTANCE_KM = 1.0   # max distance between linked points to be grouped
_ROAD_GROUP_MIN_POINTS = 2      # minimum points to form a group
_BOUNDING_BOX_PADDING = 0.0005  # ~55m padding around bounding box in degrees


def journey_disambiguate(
    query: str,
    candidates: List[DisambiguationCandidate],
    context: JourneyDisambiguationContext,
) -> JourneyDisambiguationResult:
    """
    Journey-specific disambiguation with weighted multi-signal scoring.

    Scoring formula per candidate:
        score = w1 * proximity_to_user
              + w2 * proximity_to_other_end
              + w3 * context_match          (near_area anchor)
              + w4 * name_match             (query vs candidate name)
              + w5 * user_history
              + w6 * route_feasibility

    Three possible outcomes:
        A. auto_select  — top candidate clearly better (gap >= threshold)
        B. clarify      — top 2-3 are close, ask user to pick
        C. show_list    — many similar scores or vague query, show ranked list
    """
    if not candidates:
        return JourneyDisambiguationResult(
            action="show_list", candidates=[],
            top_score=0.0,
        )

    if len(candidates) == 1:
        candidates[0].score = 1.0
        return JourneyDisambiguationResult(
            action="auto_select", candidates=candidates,
            top_score=1.0, chosen=candidates[0], confidence_gap=1.0,
        )

    # --- Score each candidate ---
    _journey_score_all(query, candidates, context)

    # Sort descending by score
    candidates.sort(key=lambda c: c.score, reverse=True)

    top_score = candidates[0].score
    gap = candidates[0].score - candidates[1].score if len(candidates) >= 2 else 1.0

    # --- Decide outcome ---
    action = "show_list"
    chosen = None

    if top_score >= _J_AUTO_SELECT_THRESHOLD and gap >= _J_AUTO_SELECT_GAP:
        action = "auto_select"
        chosen = candidates[0]
        print(
            f"[JourneyDisambig] Auto-select: '{chosen.name}' "
            f"(score={top_score:.4f}, gap={gap:.4f})"
        )
    elif top_score >= _J_SHOW_LIST_THRESHOLD and gap >= _J_AUTO_SELECT_GAP:
        # Clear winner (big gap) but below auto-select threshold — confirm with user
        action = "clarify"
        print(
            f"[JourneyDisambig] Clarify (clear winner): top={candidates[0].name} ({top_score:.4f}) "
            f"vs #{2}={candidates[1].name} ({candidates[1].score:.4f}), gap={gap:.4f}"
        )
    elif gap < _J_CLARIFY_GAP and top_score >= _J_SHOW_LIST_THRESHOLD:
        # Close race — ask user to pick between top candidates
        action = "clarify"
        print(
            f"[JourneyDisambig] Clarify (close race): top={candidates[0].name} ({top_score:.4f}) "
            f"vs #{2}={candidates[1].name} ({candidates[1].score:.4f}), gap={gap:.4f}"
        )
    else:
        action = "show_list"
        print(
            f"[JourneyDisambig] Show list: top={top_score:.4f}, "
            f"gap={gap:.4f}, {len(candidates)} candidates"
        )

    # Limit to top 5 for presentation
    presented = candidates[:5]

    return JourneyDisambiguationResult(
        action=action,
        candidates=presented,
        top_score=top_score,
        chosen=chosen,
        confidence_gap=gap,
    )


def _journey_score_all(
    query: str,
    candidates: List[DisambiguationCandidate],
    ctx: JourneyDisambiguationContext,
) -> None:
    """Score every candidate in-place using the journey scoring formula.

    Key design: signals that are *unavailable* (no geolocation, no near_area
    anchor, no user history) are excluded from the weighting entirely and
    their weight is redistributed proportionally to the remaining signals.
    This prevents missing data from dragging scores down.
    """

    q_lower = query.strip().lower()
    q_norm = DisambiguationEngine._normalize_place_name(query)

    # --- Determine which signals are available globally ---
    has_user_geo = (ctx.user_lat is not None and ctx.user_lon is not None)
    has_other_end = (ctx.other_end_lat is not None and ctx.other_end_lon is not None)
    has_context = (ctx.near_area_anchor is not None)
    has_history = (
        ctx.user_history is not None
        and len(ctx.user_history.chosen_locations) > 0
    )

    # Build {signal_name: ideal_weight} only for available signals.
    # Name match is always available. Route feasibility is always included
    # (placeholder neutral 0.5) so there's always at least two signals.
    active_weights: Dict[str, float] = {}
    active_weights["name_match"] = _JW_NAME_MATCH
    active_weights["route_feasibility"] = _JW_ROUTE_FEASIBILITY
    if has_user_geo:
        active_weights["proximity_user"] = _JW_PROXIMITY_USER
    if has_other_end:
        active_weights["proximity_other"] = _JW_PROXIMITY_OTHER
    if has_context:
        active_weights["context_match"] = _JW_CONTEXT_MATCH
    if has_history:
        active_weights["user_history"] = _JW_USER_HISTORY

    # Normalize so they sum to 1.0
    total_w = sum(active_weights.values())
    if total_w > 0:
        for k in active_weights:
            active_weights[k] /= total_w

    w_prox_user = active_weights.get("proximity_user", 0.0)
    w_prox_other = active_weights.get("proximity_other", 0.0)
    w_context = active_weights.get("context_match", 0.0)
    w_name = active_weights.get("name_match", 0.0)
    w_history = active_weights.get("user_history", 0.0)
    w_feasibility = active_weights.get("route_feasibility", 0.0)

    print(
        f"[JourneyScore] Active weights: "
        f"prox_user={w_prox_user:.3f} prox_other={w_prox_other:.3f} "
        f"context={w_context:.3f} name={w_name:.3f} "
        f"history={w_history:.3f} feasibility={w_feasibility:.3f}"
    )

    # Pre-compute max distances for normalization
    max_user_dist = 0.0
    max_other_dist = 0.0

    for c in candidates:
        if c.lat is None or c.lon is None:
            continue
        if has_user_geo:
            d = haversine_km(ctx.user_lat, ctx.user_lon, c.lat, c.lon)
            c.distance_km = d
            max_user_dist = max(max_user_dist, d)
        if has_other_end:
            d2 = haversine_km(ctx.other_end_lat, ctx.other_end_lon, c.lat, c.lon)
            max_other_dist = max(max_other_dist, d2)

    for c in candidates:
        # 1. Proximity to user (only when geolocation available)
        prox_user = 0.0
        if has_user_geo and c.lat is not None and c.lon is not None:
            d = haversine_km(ctx.user_lat, ctx.user_lon, c.lat, c.lon)
            prox_user = max(0.0, 1.0 - (d / max(max_user_dist, 0.001)))

        # 2. Proximity to other end (only when other end is resolved)
        prox_other = 0.0
        if has_other_end and c.lat is not None and c.lon is not None:
            d2 = haversine_km(ctx.other_end_lat, ctx.other_end_lon, c.lat, c.lon)
            prox_other = max(0.0, 1.0 - (d2 / max(max_other_dist, 0.001)))

        # 3. Context match — three complementary sub-signals:
        #   a) Structured: hard-boundary match on postcode prefix / borough / suburb
        #      extracted from the grounded near_area (most precise)
        #   b) Geometric:  haversine distance from grounded lat/lng anchor
        #      (fallback when structured data is absent or incomplete)
        #   c) Text:       loose token match of raw near_area text against address
        #      (catches cases the other two miss)
        # Final context_score = best single signal, boosted when multiple agree.
        context_score = 0.0
        if has_context:
            addr_text = (c.qualifier or c.name or "").lower()
            addr_tokens = set(re.findall(r"[a-z0-9]+", addr_text))

            # --- a) Structured sub-score -----------------------------------
            structured_score = 0.0
            s = ctx.near_area_structured
            if s:
                # Postcode prefix is the tightest boundary — "HA0" in address
                # means the candidate is definitely in that postcode area.
                if s.postcode_prefix:
                    # Look for the prefix as a standalone token in the address
                    pc_lower = s.postcode_prefix.lower()
                    if pc_lower in addr_tokens:
                        structured_score = max(structured_score, 1.0)

                # Borough match — candidate is in the right borough
                if s.borough and len(s.borough) > 4:
                    if s.borough.lower() in addr_text:
                        structured_score = max(structured_score, 0.90)

                # Suburb/neighbourhood match — most specific area name
                if s.suburb and len(s.suburb) > 2:
                    if s.suburb.lower() in addr_text:
                        structured_score = max(structured_score, 0.80)

            # --- b) Geometric sub-score ------------------------------------
            geo_score = 0.0
            if c.lat is not None and c.lon is not None and ctx.near_area_anchor is not None:
                anchor_dist = haversine_km(
                    ctx.near_area_anchor.lat, ctx.near_area_anchor.lng,
                    c.lat, c.lon
                )
                if anchor_dist <= 1.0:
                    geo_score = 1.0
                elif anchor_dist <= 2.0:
                    geo_score = 0.90
                elif anchor_dist <= 3.0:
                    geo_score = 0.75
                elif anchor_dist <= 5.0:
                    geo_score = 0.50
                elif anchor_dist <= 8.0:
                    geo_score = 0.20
            else:
                anchor_dist = -1.0

            # --- c) Text sub-score ----------------------------------------
            text_score = 0.0
            if ctx.near_area_text:
                near_tokens = set(re.findall(r"[a-z0-9]+", ctx.near_area_text.lower()))
                if near_tokens:
                    text_score = len(near_tokens & addr_tokens) / len(near_tokens)

            # --- Combine ---------------------------------------------------
            # Use the highest single signal as the base, then add a bonus
            # when multiple independent signals agree (convergent evidence).
            best = max(structured_score, geo_score, text_score)
            agreeing = sum(1 for v in (structured_score, geo_score, text_score) if v >= 0.5)
            if agreeing >= 2:
                context_score = min(1.0, best + 0.10)   # bonus for agreement
            else:
                context_score = best

            print(
                f"[JourneyScore] '{c.name}' "
                f"structured={structured_score:.2f} geo={geo_score:.2f} "
                f"text={text_score:.2f} → context={context_score:.2f}"
                + (f" (anchor_dist={anchor_dist:.1f}km)" if anchor_dist >= 0 else "")
            )

        # 4. Name match (always available)
        c_norm = DisambiguationEngine._normalize_place_name(c.name)
        name_sim = SequenceMatcher(None, q_norm, c_norm).ratio()
        c.name_similarity = round(name_sim, 4)

        # Also check qualifier/address for extra query tokens (e.g. area hints
        # like "paddington" that appear in the query but not the candidate name).
        addr = (c.qualifier or "").lower()
        q_tokens = set(re.findall(r"[a-z0-9]+", q_lower))
        addr_bonus = 0.0
        if addr and q_tokens:
            name_tokens = set(re.findall(r"[a-z0-9]+", (c.name or "").lower()))
            extra_tokens = q_tokens - name_tokens
            if extra_tokens:
                addr_tokens = set(re.findall(r"[a-z0-9]+", addr))
                matched = extra_tokens & addr_tokens
                addr_bonus = len(matched) / len(extra_tokens) if extra_tokens else 0.0

        # When user provides area context in the query (extra tokens matched
        # in the address), boost addr_bonus's contribution significantly so
        # that "harrow road paddington" clearly prefers Paddington results.
        if addr_bonus > 0:
            name_score = (0.50 * name_sim) + (0.50 * addr_bonus)
        else:
            name_score = name_sim

        # 5. User history (only when history data exists)
        history_score = 0.0
        if has_history and c.id:
            count = ctx.user_history.chosen_locations.get(c.id, 0)
            if count > 0:
                # Logarithmic scaling: grows quickly for first few uses, then plateaus
                history_score = min(1.0, 0.3 * math.log1p(count))

        # 6. Route feasibility (placeholder — always neutral 0.5)
        feasibility_score = 0.5

        # Weighted combination (only active signals contribute, weights sum to 1.0)
        raw_score = (
            w_prox_user * prox_user
            + w_prox_other * prox_other
            + w_context * context_score
            + w_name * name_score
            + w_history * history_score
            + w_feasibility * feasibility_score
        )

        c.score = round(min(1.0, raw_score), 4)

        print(
            f"[JourneyScore] '{c.name}' | "
            f"prox_user={prox_user:.3f} prox_other={prox_other:.3f} "
            f"context={context_score:.3f} name={name_score:.3f} "
            f"history={history_score:.3f} feasibility={feasibility_score:.3f} "
            f"| score={c.score}"
        )


def _extract_area_from_qualifier(qualifier: str) -> str:
    """Extract the area/neighbourhood name from an OSM qualifier string.

    Qualifier format is typically:
      "Road Name, Area, Borough, Greater London, England, Postcode, UK"
    We want the second comma-separated part (the area).
    """
    if not qualifier:
        return ""
    parts = [p.strip() for p in qualifier.split(",")]
    if len(parts) >= 2:
        # Skip the road name (first part), return the area (second part)
        return parts[1]
    return ""


def _group_by_road(
    candidates: List[DisambiguationCandidate],
) -> Tuple[List[RoadGroup], set]:
    """
    Group candidates that share the same road name and are in the same
    area / geographically close. Returns groups with bounding box
    coordinates for polygon display on the map, plus a set of candidate
    IDs that were absorbed into groups.

    Grouping strategy (two-pass):
      1. **Area-based**: candidates with the same normalized name AND the
         same area label (from qualifier) are grouped regardless of distance.
      2. **Proximity-based**: remaining ungrouped candidates with matching
         names are merged into an existing group if they are within
         _ROAD_GROUP_DISTANCE_KM of any member, or form a new group.

    A group needs at least _ROAD_GROUP_MIN_POINTS members.
    """
    if len(candidates) < _ROAD_GROUP_MIN_POINTS:
        return [], set()

    # Only consider candidates with coordinates
    with_coords = [c for c in candidates if c.lat is not None and c.lon is not None]
    if len(with_coords) < _ROAD_GROUP_MIN_POINTS:
        return [], set()

    # --- Pass 1: group by (normalized_name, area) ---
    # Key: (norm_name, area) → list of (index, candidate)
    area_buckets: Dict[Tuple[str, str], List[Tuple[int, DisambiguationCandidate]]] = {}
    for i, c in enumerate(with_coords):
        norm = DisambiguationEngine._normalize_place_name(c.name)
        area = _extract_area_from_qualifier(c.qualifier or "")
        key = (norm, area)
        area_buckets.setdefault(key, []).append((i, c))

    used: set = set()  # indices absorbed into groups
    groups: List[RoadGroup] = []

    for (norm_name, area), members in area_buckets.items():
        if len(members) < _ROAD_GROUP_MIN_POINTS:
            continue

        cluster_candidates = [c for _, c in members]
        cluster_indices = {i for i, _ in members}
        used.update(cluster_indices)

        lats = [c.lat for c in cluster_candidates]
        lons = [c.lon for c in cluster_candidates]

        group = RoadGroup(
            road_name=cluster_candidates[0].name,
            area=area,
            candidates=cluster_candidates,
            min_lat=min(lats) - _BOUNDING_BOX_PADDING,
            max_lat=max(lats) + _BOUNDING_BOX_PADDING,
            min_lon=min(lons) - _BOUNDING_BOX_PADDING,
            max_lon=max(lons) + _BOUNDING_BOX_PADDING,
            representative_score=max(c.score for c in cluster_candidates),
        )
        groups.append(group)
        print(
            f"[RoadGroup] '{group.road_name}' in '{area}': "
            f"{len(cluster_candidates)} points (area-based), "
            f"score={group.representative_score:.4f}"
        )

    # --- Pass 2: proximity-based for remaining candidates ---
    remaining = [
        (i, c) for i, c in enumerate(with_coords)
        if i not in used
    ]
    if len(remaining) >= _ROAD_GROUP_MIN_POINTS:
        for i, ci in remaining:
            if i in used:
                continue
            ci_norm = DisambiguationEngine._normalize_place_name(ci.name)
            cluster = [ci]
            cluster_indices = {i}

            for j, cj in remaining:
                if j in used or j == i:
                    continue
                cj_norm = DisambiguationEngine._normalize_place_name(cj.name)

                # Name must be similar
                sim = SequenceMatcher(None, ci_norm, cj_norm).ratio()
                if sim < 0.85:
                    continue

                # Must be within distance of at least one cluster member
                close = any(
                    haversine_km(ck.lat, ck.lon, cj.lat, cj.lon) <= _ROAD_GROUP_DISTANCE_KM
                    for ck in cluster
                    if ck.lat is not None and ck.lon is not None
                )
                if close:
                    cluster.append(cj)
                    cluster_indices.add(j)

            if len(cluster) >= _ROAD_GROUP_MIN_POINTS:
                used.update(cluster_indices)
                lats = [c.lat for c in cluster]
                lons = [c.lon for c in cluster]
                area = _extract_area_from_qualifier(cluster[0].qualifier or "")

                group = RoadGroup(
                    road_name=cluster[0].name,
                    area=area,
                    candidates=cluster,
                    min_lat=min(lats) - _BOUNDING_BOX_PADDING,
                    max_lat=max(lats) + _BOUNDING_BOX_PADDING,
                    min_lon=min(lons) - _BOUNDING_BOX_PADDING,
                    max_lon=max(lons) + _BOUNDING_BOX_PADDING,
                    representative_score=max(c.score for c in cluster),
                )
                groups.append(group)
                print(
                    f"[RoadGroup] '{group.road_name}' in '{area}': "
                    f"{len(cluster)} points (proximity-based), "
                    f"score={group.representative_score:.4f}"
                )

    # Collect IDs of all grouped candidates
    grouped_ids: set = set()
    for g in groups:
        for c in g.candidates:
            grouped_ids.add(c.id)

    return groups, grouped_ids


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[DisambiguationEngine] = None


def get_disambiguation_engine(sentence_model=None) -> DisambiguationEngine:
    """Get or create the singleton DisambiguationEngine.

    If the instance already exists but was created without a sentence model and
    a model is now supplied, the instance is recreated so that semantic scoring
    is enabled.  This prevents the common call-order bug where chatbot.py
    initialises the singleton first (without a model) and journey_planner.py's
    later call with a model is silently ignored.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DisambiguationEngine(sentence_model=sentence_model)
    elif sentence_model is not None and _engine_instance._sentence_model is None:
        # Upgrade: a model is now available but the existing instance lacks one.
        _engine_instance = DisambiguationEngine(sentence_model=sentence_model)
    return _engine_instance
