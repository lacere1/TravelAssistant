"""
Journey Slot Extractor – rule-based grammar patterns with optional Places grounding.

Extraction pipeline
-------------------
1.  Apply ordered grammar rules that cover every common travel phrasing.
2.  Clean each extracted candidate (strip verbs, articles, filler words).
3.  Optionally ground candidates via Google Places API (see places_grounder.py).

Grammar rules covered
---------------------
Two-slot (origin + destination):
  "from X to Y"           plan a journey from Wembley to Paddington
  "to Y from X"           get me to Oxford Circus from Neasden
  "between X and Y"       between Kilburn and Bank

Origin only:
  "from X"                journey from Wembley
  "leaving (from) X"      leaving from Wembley / leaving Wembley
  "departing (from) X"    departing from Victoria
  "starting (from) X"     starting from Harrow
  "travelling from X"     travelling from Neasden
  "I'm at X"              I'm at Paddington

Destination only:
  "to X"                  (first word after 'to' must not be a common verb)
  "arriving at/in X"      arriving at Kings Cross
  "heading to X"          heading to Brixton
  "get (me) to X"         get me to Waterloo
  "take me to X"          take me to Victoria
  "directions to X"       directions to Euston
  "navigate to X"         navigate to Canary Wharf
  "need to get to X"      I need to get to London Bridge
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Words that should never appear at the start of (or constitute all of)
# a real place name.  These are stripped from extracted candidates.
# ---------------------------------------------------------------------------
_NON_PLACE_WORDS = {
    # verbs / gerunds
    "go", "going", "get", "getting", "take", "taking", "plan", "planning",
    "travel", "travelling", "traveling", "leave", "leaving", "depart",
    "departing", "start", "starting", "head", "heading", "navigate",
    "navigating", "walk", "walking", "drive", "driving", "catch",
    "catching", "find", "see", "know", "do", "make", "be", "have",
    "want", "wanting", "need", "needing", "reach", "reaching",
    # prepositions / particles
    "from", "to", "at", "in", "on", "by", "via", "for", "of", "with",
    "into", "onto", "up", "down", "through",
    # articles / determiners / pronouns
    "a", "an", "the", "my", "your", "our", "this", "that", "these",
    "i", "im", "i'm", "me", "you", "we",
    # filler / meta
    "journey", "trip", "route", "direction", "directions", "way",
    "please", "thanks", "thank", "ok", "okay", "now", "like",
}

# Words that, when appearing immediately after 'to', signal that 'to' is
# part of an infinitive ("want to go") rather than a destination marker.
_SKIP_AFTER_TO = {
    "go", "going", "get", "getting", "take", "taking", "plan", "planning",
    "travel", "travelling", "leave", "leaving", "depart", "departing",
    "start", "starting", "head", "heading", "navigate", "navigating",
    "walk", "walking", "drive", "driving", "find", "see", "know",
    "do", "make", "be", "have", "want", "need", "reach", "check",
    "a", "an", "the", "my", "your",
}


# ---------------------------------------------------------------------------
# Grammar rules
# ---------------------------------------------------------------------------

@dataclass
class _Rule:
    """A single grammar rule for slot extraction."""
    pattern: re.Pattern
    # 1-based capture group index, or None if this rule doesn't produce that slot
    origin_group: Optional[int]
    dest_group: Optional[int]
    label: str           # human-readable name for debugging
    # If True, use reversed finditer (pick LAST match) for ambiguous patterns
    prefer_last: bool = False


_RULES: List[_Rule] = [

    # ------------------------------------------------------------------ #
    # Two-slot patterns — tried first, highest confidence                 #
    # ------------------------------------------------------------------ #

    _Rule(
        re.compile(
            r'\bfrom\s+(.+?)\s+to\s+(.+?)'
            r'(?:\s+(?:at|on|by|via|around|after|before|arriving|leaving)\b|[,.]|$)',
        ),
        origin_group=1, dest_group=2,
        label="from X to Y",
    ),
    _Rule(
        re.compile(
            r'\bto\s+(.+?)\s+from\s+(.+?)'
            r'(?:\s+(?:at|on|by|via|around|after|before)\b|[,.]|$)',
        ),
        origin_group=2, dest_group=1,
        label="to Y from X",
    ),
    _Rule(
        re.compile(
            r'\bbetween\s+(.+?)\s+and\s+(.+?)'
            r'(?:\s+(?:at|on|by|via)\b|[,.]|$)',
        ),
        origin_group=1, dest_group=2,
        label="between X and Y",
    ),

    # ------------------------------------------------------------------ #
    # Origin-only patterns                                                 #
    # ------------------------------------------------------------------ #

    _Rule(
        re.compile(
            r'\b(?:leaving|departing)\s+(?:from\s+)?(.+?)'
            r'(?:\s+(?:to|at|on|by|via)\b|[,.]|$)',
        ),
        origin_group=1, dest_group=None,
        label="leaving/departing (from) X",
    ),
    _Rule(
        re.compile(
            r'\bstarting\s+(?:from\s+)?(.+?)'
            r'(?:\s+(?:to|at|on|by|via)\b|[,.]|$)',
        ),
        origin_group=1, dest_group=None,
        label="starting (from) X",
    ),
    _Rule(
        re.compile(
            r'\btravell?ing\s+from\s+(.+?)'
            r'(?:\s+(?:to|at|on|by|via)\b|[,.]|$)',
        ),
        origin_group=1, dest_group=None,
        label="travelling from X",
    ),
    _Rule(
        re.compile(
            r"\bi(?:'m| am)\s+(?:currently\s+)?(?:at|in)\s+(.+?)"
            r'(?:\s+(?:to|at|on|by)\b|[,.]|$)',
            re.IGNORECASE,
        ),
        origin_group=1, dest_group=None,
        label="I'm at X",
    ),
    _Rule(
        re.compile(
            r'\bfrom\s+(.+?)'
            r'(?:\s+(?:at|on|by|via|around|after|before)\b|[,.]|$)',
        ),
        origin_group=1, dest_group=None,
        label="from X (origin only)",
        prefer_last=True,
    ),

    # ------------------------------------------------------------------ #
    # Destination-only patterns                                            #
    # ------------------------------------------------------------------ #

    _Rule(
        re.compile(
            r'\b(?:arriving|arrive)\s+(?:at|in|to)\s+(.+?)'
            r'(?:\s+(?:from|at|on|by)\b|[,.]|$)',
        ),
        origin_group=None, dest_group=1,
        label="arriving at Y",
    ),
    _Rule(
        re.compile(
            r'\b(?:get(?:\s+me)?|take\s+me)\s+to\s+(.+?)'
            r'(?:\s+(?:from|at|on|by)\b|[,.]|$)',
        ),
        origin_group=None, dest_group=1,
        label="get (me) to Y / take me to Y",
    ),
    _Rule(
        re.compile(
            r'\b(?:heading|going)\s+to\s+(.+?)'
            r'(?:\s+(?:from|at|on|by)\b|[,.]|$)',
        ),
        origin_group=None, dest_group=1,
        label="heading/going to Y",
    ),
    _Rule(
        re.compile(
            r'\b(?:directions|navigate|navigation)\s+to\s+(.+?)'
            r'(?:\s+(?:from|at|on|by)\b|[,.]|$)',
        ),
        origin_group=None, dest_group=1,
        label="directions/navigate to Y",
    ),
    _Rule(
        re.compile(
            r'\b(?:need|want)\s+to\s+(?:get\s+to|go\s+to|reach|arrive\s+at)\s+(.+?)'
            r'(?:\s+(?:from|at|on|by)\b|[,.]|$)',
        ),
        origin_group=None, dest_group=1,
        label="need/want to get to Y",
    ),
    _Rule(
        re.compile(
            r'\bto\s+(.+?)'
            r'(?:\s+(?:from|at|on|by|via|around|after|before)\b|[,.]|$)',
        ),
        origin_group=None, dest_group=1,
        label="to Y (destination only)",
        prefer_last=True,
    ),
]


# ---------------------------------------------------------------------------
# Candidate cleaning
# ---------------------------------------------------------------------------

def _clean(candidate: str) -> str:
    """
    Strip leading and trailing non-place words from *candidate*.

    Returns empty string if the result is entirely non-place words.

    Examples:
        "wembley"                 → "wembley"
        "oxford circus"           → "oxford circus"
        "plan a journey leaving"  → ""
        "go to victoria"          → "victoria"
        "the paddington"          → "paddington"
    """
    words = candidate.strip().split()

    # Strip leading non-place words
    while words and words[0].lower().strip(",.") in _NON_PLACE_WORDS:
        words.pop(0)

    # Strip trailing non-place words
    while words and words[-1].lower().strip(",.") in _NON_PLACE_WORDS:
        words.pop()

    result = " ".join(words).strip()

    # Reject entirely if all remaining words are non-place words
    if result and all(w.lower().strip(",.") in _NON_PLACE_WORDS for w in result.split()):
        return ""

    return result


def _restore_case(original: str, cleaned_lower: str) -> str:
    """Return the original-cased substring of *original* that matches *cleaned_lower*."""
    idx = original.lower().find(cleaned_lower.lower())
    if idx != -1:
        return original[idx: idx + len(cleaned_lower)].strip()
    return cleaned_lower  # fallback: return as-is


def _is_skip_after_to(candidate: str) -> bool:
    """Return True if candidate's first word signals an infinitive 'to + verb'."""
    first = candidate.split()[0].lower() if candidate.split() else ""
    return first in _SKIP_AFTER_TO


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class JourneySlots:
    """Structured result of slot extraction."""
    origin: Optional[str] = None          # raw extracted text (user's casing)
    destination: Optional[str] = None
    origin_near_area: Optional[str] = None     # "near X" context for origin
    destination_near_area: Optional[str] = None  # "near X" context for destination
    origin_grounded: Optional[Dict] = None    # from Places API (if available)
    destination_grounded: Optional[Dict] = None
    rule_matched: str = ""                 # label of the rule that fired (debug)

    def to_dict(self) -> Dict:
        """Backwards-compatible dict format expected by ner_processor / journey_planner."""
        out: Dict = {}
        if self.origin:
            out["origin"] = self.origin
        if self.destination:
            out["destination"] = self.destination
        if self.origin_near_area:
            out["origin_near_area"] = self.origin_near_area
        if self.destination_near_area:
            out["destination_near_area"] = self.destination_near_area
        if self.origin_grounded:
            out["origin_grounded"] = self.origin_grounded
        if self.destination_grounded:
            out["destination_grounded"] = self.destination_grounded
        return out


# ---------------------------------------------------------------------------
# Near-area context extraction
# ---------------------------------------------------------------------------

# Matches "(near X)", "(in X)", "(by X)", "(around X)" — parenthesised form
_NEAR_AREA_PAREN_RE = re.compile(
    r'\s*\(\s*(?:near|in|by|around|close\s+to)\s+(.+?)\s*\)',
    re.IGNORECASE,
)

# Matches "near X", "in X", "by X", "around X" — non-parenthesised, at end
_NEAR_AREA_SUFFIX_RE = re.compile(
    r'\s+(?:near|by|around|close\s+to)\s+(.+?)$',
    re.IGNORECASE,
)

# "in X" suffix — more cautious: only match if what follows looks like
# an area name (capitalised word or 2+ words), to avoid stripping "in" from
# place names like "Stratford International"
_NEAR_AREA_IN_SUFFIX_RE = re.compile(
    r'\s+in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)$',
)


def _extract_near_area(location: str) -> Tuple[str, Optional[str]]:
    """
    Strip "near X" / "(near X)" / "in X" / "(in X)" context from a location string.

    Returns (cleaned_location, near_area_or_None).

    Examples:
        "Harrow Road (near Sudbury)"    → ("Harrow Road", "Sudbury")
        "Lavender Avenue near Wembley"  → ("Lavender Avenue", "Wembley")
        "Harrow Road in Paddington"     → ("Harrow Road", "Paddington")
        "Oxford Circus"                 → ("Oxford Circus", None)
        "Harrow Road (near sudbury town)" → ("Harrow Road", "sudbury town")
    """
    if not location:
        return (location, None)

    # 1. Try parenthesised form first — highest confidence
    m = _NEAR_AREA_PAREN_RE.search(location)
    if m:
        area = m.group(1).strip()
        cleaned = location[:m.start()] + location[m.end():]
        cleaned = cleaned.strip()
        if cleaned and area:
            return (cleaned, area)

    # 2. Try non-parenthesised suffix: "near/by/around X"
    m = _NEAR_AREA_SUFFIX_RE.search(location)
    if m:
        area = m.group(1).strip()
        cleaned = location[:m.start()].strip()
        if cleaned and area:
            return (cleaned, area)

    # 3. Try "in X" suffix (more cautious)
    m = _NEAR_AREA_IN_SUFFIX_RE.search(location)
    if m:
        area = m.group(1).strip()
        cleaned = location[:m.start()].strip()
        if cleaned and area:
            return (cleaned, area)

    return (location, None)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class JourneySlotExtractor:
    """
    Rule-based journey slot extractor.

    Uses ordered grammar rules to find ORIGIN / DESTINATION candidates,
    then optionally grounds them via Google Places API.
    """

    def __init__(self):
        self._ready = True
        # Lazy-loaded Places grounder (only initialised if API key available)
        self._grounder = None
        self._grounder_checked = False

    @property
    def ready(self) -> bool:
        return self._ready

    def _get_grounder(self):
        """Lazy-load the Places grounder once."""
        if not self._grounder_checked:
            self._grounder_checked = True
            try:
                from places_grounder import get_grounder
                g = get_grounder()
                if g.available:
                    self._grounder = g
                    print("[JourneySlotExtractor] Places grounding enabled.")
                else:
                    print("[JourneySlotExtractor] No Places API key found – grounding disabled.")
            except ImportError:
                pass
        return self._grounder

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def _apply_rule(self, rule: _Rule, text: str, text_lower: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Try to apply *rule* to *text_lower*.

        Returns (origin_candidate, dest_candidate) where each is either a
        cleaned string or None.  Returns None if the rule doesn't match or
        both candidates are empty after cleaning.

        All matches are tried (via finditer) so that if an early match is
        rejected (e.g. "to go …" skipped), later matches still get a chance.
        prefer_last reverses the iteration order.
        """
        # Use overlapping search (advance by 1 each time) so that a pattern
        # like "to Y from X" finds BOTH "to go to Oxford Circus from Neasden"
        # AND "to Oxford Circus from Neasden" — the first is then skipped
        # via _is_skip_after_to, the second is used.
        matches = []
        pos = 0
        while pos < len(text_lower):
            m = rule.pattern.search(text_lower, pos)
            if not m:
                break
            matches.append(m)
            pos = m.start() + 1  # advance by 1 to allow overlapping

        if not matches:
            return None

        matches_to_try = list(reversed(matches)) if rule.prefer_last else matches

        for m in matches_to_try:
            origin_raw = m.group(rule.origin_group) if rule.origin_group else None
            dest_raw = m.group(rule.dest_group) if rule.dest_group else None

            # Skip "to X" if X starts with a verb
            if dest_raw and _is_skip_after_to(dest_raw):
                continue

            origin_cleaned = _clean(origin_raw) if origin_raw else None
            dest_cleaned = _clean(dest_raw) if dest_raw else None

            # Require at least one valid slot for the rule to count
            if origin_cleaned or dest_cleaned:
                # Restore original casing
                if origin_cleaned:
                    origin_cleaned = _restore_case(text, origin_cleaned)
                if dest_cleaned:
                    dest_cleaned = _restore_case(text, dest_cleaned)
                return (origin_cleaned, dest_cleaned)

        return None

    def extract(self, text: str, ground: bool = True) -> JourneySlots:
        """
        Full extraction pipeline: grammar rules → cleaning → optional grounding.

        Args:
            text:   The user's raw message.
            ground: Whether to call Places API to ground results (default True).

        Returns:
            A :class:`JourneySlots` instance.  Fields are None when not found.
        """
        result = JourneySlots()

        # Pre-pass: strip ALL parenthesised near-area clauses from the text
        # BEFORE applying grammar rules so that "(close to X)" or "(near X)"
        # don't interfere with "from … to …" splitting.  We track each
        # extracted area with its *character position* in the original text
        # so we can later assign it to the correct slot.
        pre_areas: List[Tuple[int, str]] = []     # (char_pos, area_text)
        text_stripped = text

        def _collect_paren(m: re.Match) -> str:
            pre_areas.append((m.start(), m.group(1).strip()))
            return ""                    # remove from text

        text_stripped = _NEAR_AREA_PAREN_RE.sub(_collect_paren, text_stripped)
        text_stripped = re.sub(r'\s{2,}', ' ', text_stripped).strip()

        text_lower = text_stripped.lower()

        # Try two-slot rules first, then single-slot rules
        two_slot = [r for r in _RULES if r.origin_group and r.dest_group]
        one_slot = [r for r in _RULES if not (r.origin_group and r.dest_group)]

        for rule in two_slot + one_slot:
            slots = self._apply_rule(rule, text_stripped, text_lower)
            if slots is None:
                continue

            origin, dest = slots

            if origin and not result.origin:
                # Extract any remaining suffix-form "near X" context from origin
                origin_clean, origin_area = _extract_near_area(origin)
                result.origin = origin_clean
                if origin_area:
                    result.origin_near_area = origin_area
            if dest and not result.destination:
                # Extract any remaining suffix-form "near X" context from dest
                dest_clean, dest_area = _extract_near_area(dest)
                result.destination = dest_clean
                if dest_area:
                    result.destination_near_area = dest_area
            if result.origin or result.destination:
                result.rule_matched = rule.label
                break  # stop on first rule that produces anything

        # Assign pre-extracted parenthesised areas to origin/dest slots.
        # Each area is matched to the slot whose text appeared just before it
        # in the original input (by character position).
        if pre_areas:
            text_lower_orig = text.lower()
            origin_pos = text_lower_orig.find(result.origin.lower()) if result.origin else -1
            dest_pos = text_lower_orig.find(result.destination.lower()) if result.destination else -1

            for area_pos, area_text in pre_areas:
                # Determine which slot this area belongs to by proximity:
                # it belongs to the slot whose text ends just before it.
                origin_end = (origin_pos + len(result.origin)) if result.origin and origin_pos >= 0 else -999
                dest_end = (dest_pos + len(result.destination)) if result.destination and dest_pos >= 0 else -999

                # The area follows whichever slot ended most recently before it
                assign_to_origin = (
                    origin_end <= area_pos
                    and (dest_end > area_pos or origin_end > dest_end)
                )
                assign_to_dest = (
                    dest_end <= area_pos
                    and (origin_end > area_pos or dest_end > origin_end)
                )

                if assign_to_origin and result.origin and not result.origin_near_area:
                    result.origin_near_area = area_text
                elif assign_to_dest and result.destination and not result.destination_near_area:
                    result.destination_near_area = area_text
                elif result.origin and not result.origin_near_area:
                    result.origin_near_area = area_text
                elif result.destination and not result.destination_near_area:
                    result.destination_near_area = area_text

        # Optional Places API grounding
        if ground and (result.origin or result.destination):
            grounder = self._get_grounder()
            if grounder:
                if result.origin:
                    result.origin_grounded = grounder.ground(result.origin)
                if result.destination:
                    result.destination_grounded = grounder.ground(result.destination)

        if result.origin or result.destination:
            print(
                f"[JourneySlotExtractor] Rule='{result.rule_matched}' "
                f"origin={result.origin!r} (near={result.origin_near_area!r}) "
                f"dest={result.destination!r} (near={result.destination_near_area!r})"
            )

        return result

    def extract_journey_slots(self, text: str) -> Dict:
        """
        Backwards-compatible wrapper — returns a plain dict.

        Keys: 'origin', 'destination', 'origin_grounded', 'destination_grounded'
        (only keys that have values are included).
        """
        return self.extract(text).to_dict()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_singleton: Optional[JourneySlotExtractor] = None


def get_extractor() -> JourneySlotExtractor:
    """Return (and lazily create) the singleton extractor."""
    global _singleton
    if _singleton is None:
        _singleton = JourneySlotExtractor()
    return _singleton

