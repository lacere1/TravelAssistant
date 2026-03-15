"""
Transport Data Fetcher
Integrates with real-world transport APIs (TfL, Google Maps).
Timetables, line status, disruptions, route recommendations, stop info.
Returns None or empty list if no API data is available (no mock data fallback)
"""
import requests
import os
import re
from typing import Dict, Optional, Any, List
import time
from datetime import datetime
from urllib.parse import quote
from difflib import SequenceMatcher

class TransportDataFetcher:
    def __init__(self):
        """Initialize transport API clients"""
        # Transport for London (TFL) API credentials
        self.tfl_app_id = os.environ.get('TFL_APP_ID', '')
        self.tfl_app_key = os.environ.get('TFL_APP_KEY', '')
        self.tfl_base_url = 'https://api.tfl.gov.uk'
        
        # Check which APIs are available
        self.has_tfl = bool(self.tfl_app_id and self.tfl_app_key)
        self.use_mock_data = not self.has_tfl
        
        if self.has_tfl:
            print("TFL API initialized (App ID and Key configured)")
        if self.use_mock_data:
            print("Warning: No API keys found. Using mock traffic data for demonstration.")
    
    def _tfl_params(self) -> Dict[str, str]:
        """Common TfL auth params. app_id is optional; app_key is typically required."""
        params = {}
        if self.tfl_app_id:
            params["app_id"] = self.tfl_app_id
        if self.tfl_app_key:
            params["app_key"] = self.tfl_app_key
        return params

    def _expand_to_leaf_bus_stops(self, stop_id: str) -> List[str]:
        """
        If stop_id is a StopPoint group/area, expand to leaf bus stop IDs.

        Returns the stop ID directly — group expansion is handled
        the id we were given so timetable logic can still call this helper
        without needing the full StopPoint hierarchy traversal.
        """
        return [stop_id] if stop_id else []
    
    def get_route_traffic(self, route: str) -> Optional[Dict[str, Any]]:
        """
        Get traffic information for a specific route
        
        Args:
            route: Route identifier (e.g., "Bakerloo", "Central", "Northern")
            
        Returns:
            dict: Traffic data including status, delay, congestion
        """
        # Try TFL API first (for London transport)
        if self.has_tfl:
            try:
                tfl_data = self._fetch_tfl_line_status(route)
                if tfl_data:
                    return tfl_data
            except Exception as e:
                print(f"TFL API error: {e}")
        
        # No API data available - return None instead of mock data
        return None
    
    def _normalize_query(self, query: str) -> tuple:
        """
        Normalize a query by removing common words and extracting key terms
        Returns: (cleaned_query, main_words, first_word)
        
        Centralized word filtering - removes common stop words and query words
        that shouldn't be part of location/route names.
        """
        # Common words to remove (stop words, query words, transport terms)
        common_words = {
            'station', 'stop', 'the', 'tube', 'underground', 'metro', 'london',
            'times', 'time', 'at', 'for', 'from', 'to', 'is', 'are', 'was', 'were',
            'what', 'where', 'when', 'how', 'why', 'which', 'like', 'traffic',
            'bus', 'train', 'next', 'timetable', 'arrival', 'arrivals'
        }
        
        if not query or not query.strip():
            return '', [], None
        
        words = [w.lower().strip() for w in query.split() if w.strip()]
        meaningful_words = [w for w in words if w not in common_words and len(w) > 1]
        cleaned = ' '.join(meaningful_words)
        main_words = [w for w in meaningful_words if len(w) > 2]
        first_word = main_words[0] if main_words else meaningful_words[0] if meaningful_words else None
        return cleaned, main_words, first_word
    
    def _normalize_tube_station_name(self, station_name: str) -> str:
        """
        Normalize punctuation and casing for typical tube station names.
        
        Handles:
        - "St." / "St" → "St" (standardize abbreviation)
        - "&" → "and"
        - Hyphens and spacing
        - Title case formatting
        - Apostrophes (preserved)
        
        Args:
            station_name: Raw station name (e.g., "baker st.", "harrow & wealdstone")
            
        Returns:
            Normalized station name (e.g., "Baker St", "Harrow and Wealdstone")
        """
        if not station_name or not station_name.strip():
            return station_name
        
        # Start with trimmed input
        normalized = station_name.strip()
        
        # Normalize "St." / "St" → "St" (standardize abbreviation without period)
        # Handle both standalone and in context (e.g., "St. James's", "St Paul's")
        normalized = re.sub(r'\bSt\.\b', 'St', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bst\b', 'St', normalized)  # Lowercase "st" → "St"
        
        # Normalize "&" → "and"
        normalized = re.sub(r'\s*&\s*', ' and ', normalized)
        
        # Normalize hyphens - ensure single hyphen with spaces around if needed
        # But preserve hyphens in compound names (e.g., "Bromley-by-Bow")
        normalized = re.sub(r'\s*-\s*', '-', normalized)  # Remove spaces around hyphens
        
        # Normalize multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Apply title case, but preserve special cases:
        # - "St" should remain capitalized
        # - Words after apostrophes should be capitalized (e.g., "James's")
        # - Words after hyphens should be capitalized (e.g., "by-Bow")
        # - Small words like "and", "on", "the" in the middle should be lowercase
        #   (but capitalize if first word)
        
        words = normalized.split()
        if not words:
            return normalized
        
        # Title case each word, with special handling
        title_words = []
        small_words = {'and', 'on', 'the', 'of', 'by', 'in', 'at', 'to', 'for'}
        
        for i, word in enumerate(words):
            # Handle hyphenated words
            if '-' in word:
                parts = word.split('-')
                title_parts = []
                for j, part in enumerate(parts):
                    if part.lower() == 'st':
                        title_parts.append('St')
                    elif j == 0 or part.lower() not in small_words:
                        title_parts.append(part.capitalize())
                    else:
                        title_parts.append(part.lower())
                title_words.append('-'.join(title_parts))
            # Handle words with apostrophes
            elif "'" in word:
                # Split on apostrophe, capitalize first part, handle second part
                if word.lower().startswith("st'"):
                    # Special case: "St's" → "St's"
                    title_words.append("St" + word[2:].capitalize())
                else:
                    parts = word.split("'")
                    if len(parts) == 2:
                        title_words.append(parts[0].capitalize() + "'" + parts[1].capitalize())
                    else:
                        title_words.append(word.capitalize())
            # Handle "St" abbreviation
            elif word.lower() == 'st':
                title_words.append('St')
            # Handle regular words
            else:
                if i == 0:
                    # First word always capitalized
                    title_words.append(word.capitalize())
                elif word.lower() in small_words:
                    # Small words in middle are lowercase
                    title_words.append(word.lower())
                else:
                    title_words.append(word.capitalize())
        
        normalized = ' '.join(title_words)
        
        # Final cleanup: remove any extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        Efficient word similarity calculation handling typos, extra chars, missing chars
        """
        w1, w2 = word1.lower(), word2.lower()
        if w1 == w2:
            return 1.0
        
        # Direct similarity (handles most typos)
        direct = SequenceMatcher(None, w1, w2).ratio()
        
        # Quick checks for early exit
        if direct > 0.7:
            return direct
        
        # Character set overlap (handles scrambled letters)
        chars1, chars2 = set(w1), set(w2)
        char_overlap = len(chars1 & chars2) / max(len(chars1 | chars2), 1) if chars1 | chars2 else 0
        
        # First/last character bonus (common typo patterns)
        first_match = 0.15 if (w1 and w2 and w1[0] == w2[0]) else 0
        last_match = 0.15 if (len(w1) > 1 and len(w2) > 1 and w1[-1] == w2[-1]) else 0
        
        # Length similarity (handles extra/missing characters)
        len_sim = 1.0 - abs(len(w1) - len(w2)) / max(len(w1), len(w2), 1)
        
        # Substring match (one word contains the other - handles partial matches)
        substring_bonus = 0.3 if (w1 in w2 or w2 in w1) and min(len(w1), len(w2)) >= 3 else 0
        
        # Combined score
        return max(direct, char_overlap * 0.6) + first_match + last_match + (len_sim * 0.1) + substring_bonus
    
    def _find_closest_location_match(self, location: str, all_stops: List[Dict[str, Any]]) -> Optional[str]:
        """
        Efficiently find closest matching stop using optimized fuzzy matching
        Handles typos, extra chars, missing chars, word order changes
        """
        if not all_stops:
            return None
        
        # Normalize the query once
        cleaned_loc, main_words, first_word = self._normalize_query(location)
        location_lower = location.lower()
        
        if not main_words and not cleaned_loc:
            return None
        
        best_match = None
        best_score = 0.0
        
        # Pre-process stops for efficiency
        for stop in all_stops:
            stop_name = stop.get('name', '')
            if not stop_name:
                continue
            
            stop_lower = stop_name.lower()
            cleaned_stop, stop_words, _ = self._normalize_query(stop_name)
            
            # Calculate multiple similarity scores efficiently
            scores = []
            
            # 1. Full string similarity (exact or close matches)
            full_match = SequenceMatcher(None, location_lower, stop_lower).ratio()
            scores.append(full_match)
            
            # 2. Cleaned string similarity (removes common words)
            if cleaned_loc and cleaned_stop:
                cleaned_match = SequenceMatcher(None, cleaned_loc, cleaned_stop).ratio()
                scores.append(cleaned_match * 0.98)
            
            # 3. Word-level matching (handles word order and typos)
            if main_words and stop_words:
                # Best match for each location word in stop words
                word_scores = []
                for loc_word in main_words:
                    best_match_score = max(
                        self._calculate_word_similarity(loc_word, stop_word)
                        for stop_word in stop_words
                    )
                    word_scores.append(best_match_score)
                
                # Average word match score
                avg_word_score = sum(word_scores) / len(word_scores) if word_scores else 0
                scores.append(avg_word_score * 0.95)
                
                # Coverage: how many location words found in stop
                coverage = sum(1 for score in word_scores if score > 0.6) / len(word_scores) if word_scores else 0
                scores.append(coverage * 0.85)
            
            # 4. Character-level similarity (handles scrambled words)
            loc_chars = set(location_lower.replace(' ', '').replace('-', ''))
            stop_chars = set(stop_lower.replace(' ', '').replace('-', ''))
            if loc_chars and stop_chars:
                char_sim = len(loc_chars & stop_chars) / max(len(loc_chars | stop_chars), 1)
                scores.append(char_sim * 0.5)
            
            # 5. Substring/contains match (handles partial matches)
            if len(location_lower) >= 3:
                if location_lower in stop_lower or stop_lower in location_lower:
                    scores.append(0.75)
            
            # 6. First word emphasis (first word is usually most important)
            if first_word and stop_words:
                first_word_sim = max(
                    self._calculate_word_similarity(first_word, sw)
                    for sw in stop_words
                )
                scores.append(first_word_sim * 0.9)
            
            # Use maximum score (best match across all methods)
            final_score = max(scores) if scores else 0
            
            if final_score > best_score:
                best_score = final_score
                best_match = stop_name
        
        # Threshold: 0.3 for serious typos (30% similarity)
        return best_match if best_score > 0.3 else None
    
    def get_tfl_timetable(self, stop_query: str, mode_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed timetable for a TFL stop point, grouped by mode (buses/trains)
        
        Args:
            stop_query: TFL stop point name or description (e.g., "Oxford Circus", "Wembley Central Station")
            mode_filter: Optional filter - 'bus', 'train', or None for both
            
        Returns:
            dict: Timetable data with buses and trains separately
        """
        if not self.has_tfl:
            return None
        
        try:
            # Extract bus route numbers from query early (e.g. "83", "302", "N83")
            # This allows us to search for just the stop name but filter by route later
            bus_route_pattern = r'\b([Nn]?\d{1,3})\b'
            route_matches = re.findall(bus_route_pattern, stop_query)
            mentioned_routes = [r.upper() for r in route_matches]  # Normalize to uppercase
            
            # Remove route numbers from search query (TFL search works better with just stop name)
            # e.g. "Lavender Avenue 83" -> "Lavender Avenue"
            search_query = stop_query
            if mentioned_routes:
                # Remove route numbers from the query for TFL search
                for route in mentioned_routes:
                    # Remove the route number and any "route" keyword before it
                    search_query = re.sub(r'\b(?:route\s+)?' + re.escape(route) + r'\b', '', search_query, flags=re.IGNORECASE)
                search_query = ' '.join(search_query.split())  # Clean up extra spaces
            
            # For train/tube queries, normalize punctuation/casing for typical tube station names
            if mode_filter == 'train':
                search_query = self._normalize_tube_station_name(search_query)
            
            # Search for stop points matching the user query
            url = f"{self.tfl_base_url}/StopPoint/Search"
            
            # If mode_filter is 'train' and query contains "station", try base name first
            # This helps find "Underground Station" entries when user says just "station"
            initial_query = search_query
            if mode_filter == 'train' and 'station' in stop_query.lower() and 'underground' not in stop_query.lower():
                # Try base name first (without "station") for better train station matching
                common_words = ['station', 'stop', 'the', 'tube', 'underground', 'metro']
                query_words = stop_query.split()
                cleaned_query = ' '.join(w for w in query_words if w.lower() not in common_words).strip()
                if cleaned_query:
                    initial_query = cleaned_query
            
            params = {
                'app_id': self.tfl_app_id,
                'app_key': self.tfl_app_key,
                'query': initial_query,
                'modes': 'tube,bus,dlr,overground,tram'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check if any stop points were found
            matches = data.get('matches', [])
            
            # If mode_filter is 'train', check if we have any train/tube stations in results
            # If not, try variations with "Underground Station" or just the base name
            if matches and mode_filter == 'train':
                has_train_station = any(
                    any(m in match.get('modes', []) for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])
                    for match in matches
                )
                if not has_train_station:
                    # No train stations found, try variations
                    query_lower = stop_query.lower()
                    
                    # Strategy 1: Try just the base name (remove "station") - this often works better
                    common_words = ['station', 'stop', 'the', 'tube', 'underground', 'metro']
                    query_words = stop_query.split()
                    cleaned_query = ' '.join(w for w in query_words if w.lower() not in common_words).strip()
                    if cleaned_query and cleaned_query.lower() != stop_query.lower():
                        params['query'] = cleaned_query
                        try:
                            response = requests.get(url, params=params, timeout=10)
                            if response.status_code == 200:
                                cleaned_data = response.json()
                                cleaned_matches = cleaned_data.get('matches', [])
                                if cleaned_matches:
                                    # Check if we now have train stations
                                    has_train = any(
                                        any(m in match.get('modes', []) for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])
                                        for match in cleaned_matches
                                    )
                                    if has_train:
                                        matches = cleaned_matches
                        except Exception:
                            pass
                    
                    # Strategy 2: Try replacing "station" with "Underground Station"
                    if not matches or not any(any(m in match.get('modes', []) for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail']) for match in matches):
                        if 'station' in query_lower and 'underground' not in query_lower:
                            # Try replacing "station" with "Underground Station"
                            underground_query = stop_query.replace('station', 'Underground Station').replace('Station', 'Underground Station')
                            underground_query = self._normalize_tube_station_name(underground_query)
                            params['query'] = underground_query
                            try:
                                response = requests.get(url, params=params, timeout=10)
                                if response.status_code == 200:
                                    underground_data = response.json()
                                    underground_matches = underground_data.get('matches', [])
                                    if underground_matches:
                                        matches = underground_matches
                            except Exception:
                                pass
            
            if not matches or len(matches) == 0:
                # Try with cleaned version of the query (remove common words)
                common_words = ['station', 'stop', 'the', 'tube', 'underground', 'metro']
                query_words = stop_query.split()
                cleaned_query = ' '.join(w for w in query_words if w.lower() not in common_words).strip()
                
                if cleaned_query and cleaned_query.lower() != stop_query.lower():
                    # Try search with cleaned query
                    params['query'] = cleaned_query
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        matches = data.get('matches', [])
                
                # If mode_filter is 'train' and still no matches, try adding "Underground Station"
                if (not matches or len(matches) == 0) and mode_filter == 'train':
                    query_lower = stop_query.lower()
                    if 'station' in query_lower or 'underground' not in query_lower:
                        # Try adding "Underground Station" if not already present
                        if 'underground' not in query_lower:
                            underground_query = stop_query.replace('station', 'Underground Station').replace('Station', 'Underground Station')
                            underground_query = self._normalize_tube_station_name(underground_query)
                            if underground_query != stop_query:
                                params['query'] = underground_query
                                try:
                                    response = requests.get(url, params=params, timeout=10)
                                    if response.status_code == 200:
                                        underground_data = response.json()
                                        underground_matches = underground_data.get('matches', [])
                                        if underground_matches:
                                            matches = underground_matches
                                except Exception:
                                    pass
                
                # If still no matches, try multiple broader search strategies for serious typos
                if not matches or len(matches) == 0:
                    main_words = [w for w in stop_query.split() if len(w) > 2 and w.lower() not in common_words]
                    
                    # Strategy: Use first main word only (broad)
                    if main_words:
                        broader_query = main_words[0]
                        params['query'] = broader_query
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 200:
                            broader_data = response.json()
                            broader_matches = broader_data.get('matches', [])
                            if broader_matches:
                                closest_match = self._find_closest_location_match(stop_query, broader_matches)
                                if closest_match:
                                    params['query'] = closest_match
                                    response = requests.get(url, params=params, timeout=10)
                                    if response.status_code == 200:
                                        data = response.json()
                                        matches = data.get('matches', [])
            
            # Check if any stop points were found after retry attempts
            if not matches or len(matches) == 0:
                return {'error': 'not_found', 'query': stop_query}
            
            # Get arrivals from all matching stops
            # For buses: fetch ALL matching bus stops (e.g., "Oxford Circus / Margaret Street", "Oxford Circus / Regent Street")
            # For trains: fetch from all matching stations (though usually there's just one)
            all_arrivals = []
            stop_names = set()
            stop_ids_processed = set()  # Avoid duplicate API calls
            
            # If mode_filter is 'train', prioritize train/tube stations
            # Sort matches to prioritize train stations first
            if mode_filter == 'train':
                def train_priority(match):
                    stop_modes = match.get('modes', [])
                    has_train = any(m in stop_modes for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])
                    # Prioritize matches with "Underground Station" in name for train queries
                    name = match.get('name', '').lower()
                    is_underground_station = 'underground station' in name
                    return (has_train, is_underground_station)
                matches = sorted(matches, key=train_priority, reverse=True)
                
                # Filter to only train stations for disambiguation check
                train_station_matches = [
                    match for match in matches
                    if any(m in match.get('modes', []) for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])
                ]
                
                # If multiple train stations found, check if user's query uniquely matches one
                if len(train_station_matches) > 1:
                    seen_names = set()
                    unique_station_names = []
                    disambiguation_options = []
                    for match in train_station_matches:
                        raw_name = match.get('name', 'Unknown')
                        name = self._normalize_tube_station_name(raw_name)
                        if name not in seen_names:
                            seen_names.add(name)
                            unique_station_names.append(name)
                            disambiguation_options.append({
                                'id': match.get('id', ''),
                                'name': raw_name,
                                'label': name,
                            })
                    # If user's query uniquely matches one option (e.g. "Finchley Road and Frognal Rail Station"),
                    # use it and skip disambiguation
                    stop_lower = stop_query.strip().lower()
                    matching_opts = [
                        opt for opt in disambiguation_options
                        if (opt.get('label') or opt.get('name') or '').lower()
                        and ((opt.get('label') or opt.get('name') or '').lower() in stop_lower
                             or stop_lower in (opt.get('label') or opt.get('name') or '').lower())
                    ]
                    if len(matching_opts) == 1:
                        matches = [m for m in train_station_matches if m.get('id') == matching_opts[0]['id']]
                    else:
                        # Zero or multiple matches; return disambiguation
                        return {
                            'error': 'disambiguation_needed',
                            'query': stop_query,
                            'mode': 'train',
                            'stations': unique_station_names,
                            'count': len(unique_station_names),
                            'disambiguation_options': disambiguation_options
                        }
            
            # If mode_filter is 'bus', check for multiple distinct bus stop locations
            if mode_filter == 'bus':
                # Filter to only bus stops for disambiguation check
                bus_stop_matches = [
                    match for match in matches
                    if 'bus' in match.get('modes', [])
                ]
                
                # Use the route numbers we extracted at the start of the function
                # (Routes were already extracted from stop_query and stored in mentioned_routes)
                route_was_specified = len(mentioned_routes) > 0
                
                # If user mentioned a bus route, filter stops to only those serving that route
                if mentioned_routes and len(bus_stop_matches) >= 1:
                    filtered_stops = []
                    for match in bus_stop_matches:
                        stop_lines = match.get('lines', [])
                        # Check if any mentioned route is in this stop's lines
                        stop_line_ids = []
                        for line in stop_lines:
                            if isinstance(line, dict):
                                line_id = line.get('id', '').upper()
                            else:
                                line_id = str(line).upper()
                            stop_line_ids.append(line_id)
                        
                        # Also check if this is a group stop - need to check child stops
                        stop_id = match.get('id', '')
                        serves_route = any(route in stop_line_ids for route in mentioned_routes)
                        
                        # If group stop doesn't have the route in its own lines, check children
                        if not serves_route and stop_id:
                            try:
                                stop_info_url = f"{self.tfl_base_url}/StopPoint/{stop_id}"
                                info_params = {
                                    'app_id': self.tfl_app_id,
                                    'app_key': self.tfl_app_key,
                                }
                                info_resp = requests.get(stop_info_url, params=info_params, timeout=3)
                                if info_resp.status_code == 200:
                                    info_data = info_resp.json()
                                    children = info_data.get('children', [])
                                    for child in children:
                                        child_lines = child.get('lines', [])
                                        for line in child_lines:
                                            if isinstance(line, dict):
                                                line_id = line.get('id', '').upper()
                                            else:
                                                line_id = str(line).upper()
                                            if line_id in mentioned_routes:
                                                serves_route = True
                                                break
                                        if serves_route:
                                            break
                            except Exception:
                                pass  # If check fails, include the stop anyway
                        
                        if serves_route:
                            filtered_stops.append(match)
                    
                    # If user specified a route, use all stops that serve that route (no disambiguation)
                    if len(filtered_stops) > 0:
                        # User specified a route - return all stops serving that route
                        bus_stop_matches = filtered_stops
                        # Update matches to only include filtered bus stops (plus any train stops if mode_filter allows)
                        # This ensures the main loop only processes stops serving the specified route
                        if mode_filter == 'bus':
                            # Replace matches with only the filtered bus stops
                            matches = filtered_stops
                        else:
                            # Keep train stops, but filter bus stops
                            train_matches = [m for m in matches if any(mode in m.get('modes', []) for mode in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])]
                            matches = filtered_stops + train_matches
                        # Skip disambiguation - proceed to fetch arrivals for all stops serving the route
                    elif route_was_specified and len(filtered_stops) == 0:
                        # User specified a route but no stops serve that route - return error with available routes
                        available_routes = set()
                        
                        # Collect all routes from all bus stops (including children for group stops)
                        for match in bus_stop_matches:
                            stop_lines = match.get('lines', [])
                            for line in stop_lines:
                                if isinstance(line, dict):
                                    line_id = line.get('id', '')
                                else:
                                    line_id = str(line)
                                if line_id:
                                    available_routes.add(line_id)
                            
                            # Also check child stops for group stops
                            stop_id = match.get('id', '')
                            if stop_id:
                                try:
                                    stop_info_url = f"{self.tfl_base_url}/StopPoint/{stop_id}"
                                    info_params = {
                                        'app_id': self.tfl_app_id,
                                        'app_key': self.tfl_app_key,
                                    }
                                    info_resp = requests.get(stop_info_url, params=info_params, timeout=3)
                                    if info_resp.status_code == 200:
                                        info_data = info_resp.json()
                                        children = info_data.get('children', [])
                                        for child in children:
                                            child_lines = child.get('lines', [])
                                            for line in child_lines:
                                                if isinstance(line, dict):
                                                    line_id = line.get('id', '')
                                                else:
                                                    line_id = str(line)
                                                if line_id:
                                                    available_routes.add(line_id)
                                except Exception:
                                    pass
                        
                        # Format available routes for display
                        sorted_routes = sorted(available_routes, key=lambda x: (len(x), x))
                        routes_str = ', '.join(sorted_routes) if sorted_routes else 'none'
                        
                        # Get the stop name for the error message
                        stop_name = bus_stop_matches[0].get('name', stop_query) if bus_stop_matches else stop_query
                        requested_route = ', '.join(mentioned_routes)
                        
                        return {
                            'error': 'route_not_served',
                            'query': stop_query,
                            'stop_name': stop_name,
                            'requested_route': requested_route,
                            'available_routes': sorted_routes,
                            'available_routes_str': routes_str
                        }
                
                # Only check for disambiguation if user did NOT specify a route
                # If route was specified, we already filtered and will return all matching stops
                
                # If multiple bus stops found and no route specified, check for disambiguation
                if len(bus_stop_matches) > 1 and not route_was_specified:
                    # For grouped bus stops (e.g. 490G...), break them down into individual child stops
                    # so each platform/direction becomes its own disambiguation option.
                    expanded_matches = []
                    for match in bus_stop_matches:
                        stop_id = match.get('id', '')
                        name = match.get('name', stop_query)
                        # Heuristic: group StopPoints often have 4th char 'G' (e.g. 490G...)
                        is_group = bool(stop_id and len(stop_id) >= 4 and stop_id[3] == 'G')
                        if not is_group:
                            expanded_matches.append(match)
                            continue
                        try:
                            stop_info_url = f"{self.tfl_base_url}/StopPoint/{stop_id}"
                            info_resp = requests.get(stop_info_url, params=self._tfl_params(), timeout=5)
                            if info_resp.status_code == 200:
                                info_data = info_resp.json()
                                children = info_data.get('children', [])
                                child_added = False
                                for child in children:
                                    if 'bus' not in (child.get('modes') or []):
                                        continue
                                    child_match = {
                                        'id': child.get('id', stop_id),
                                        'name': name,  # keep base location name
                                        'towards': child.get('towards') or '',
                                        'lat': child.get('lat'),
                                        'lon': child.get('lon'),
                                        'modes': child.get('modes') or ['bus'],
                                        'icsId': child.get('icsId', match.get('icsId')),
                                    }
                                    expanded_matches.append(child_match)
                                    child_added = True
                                # If no child bus stops were found, keep the original group match
                                if not child_added:
                                    expanded_matches.append(match)
                            else:
                                expanded_matches.append(match)
                        except Exception:
                            expanded_matches.append(match)
                    if expanded_matches:
                        bus_stop_matches = expanded_matches
                    # Collect all coordinates from all bus stops
                    lats = [m.get('lat') for m in bus_stop_matches if m.get('lat') is not None]
                    lons = [m.get('lon') for m in bus_stop_matches if m.get('lon') is not None]
                    
                    needs_disambiguation = False
                    
                    # Grouped stops: same station name (or icsId) with different "towards" = direction disambiguation
                    towards_set = set((m.get('towards') or '').strip() for m in bus_stop_matches if (m.get('towards') or '').strip())
                    if len(towards_set) > 1:
                        needs_disambiguation = True
                    
                    # Check if all stops have coordinates (for location-based disambiguation)
                    if not needs_disambiguation and lats and lons and len(lats) == len(bus_stop_matches) and len(lons) == len(bus_stop_matches):
                        lat_diff = max(lats) - min(lats) if len(lats) > 1 else 0
                        lon_diff = max(lons) - min(lons) if len(lons) > 1 else 0
                        if lat_diff >= 0.05 or lon_diff >= 0.05:
                            needs_disambiguation = True
                    elif not needs_disambiguation:
                        # If coordinates are missing, fall back to name-based disambiguation
                        # Group by exact name first
                        stops_by_exact_name = {}
                        for match in bus_stop_matches:
                            name = match.get('name', 'Unknown')
                            if name not in stops_by_exact_name:
                                stops_by_exact_name[name] = []
                            stops_by_exact_name[name].append(match)
                        
                        # Check if any stops with same exact name have different coordinates
                        for name, name_matches in stops_by_exact_name.items():
                            if len(name_matches) > 1:
                                # Multiple stops with same exact name - check coordinates
                                name_lats = [m.get('lat') for m in name_matches if m.get('lat') is not None]
                                name_lons = [m.get('lon') for m in name_matches if m.get('lon') is not None]
                                
                                if name_lats and name_lons:
                                    # Check if lat or lon difference is >= 0.05
                                    name_lat_diff = max(name_lats) - min(name_lats) if len(name_lats) > 1 else 0
                                    name_lon_diff = max(name_lons) - min(name_lons) if len(name_lons) > 1 else 0
                                    
                                    if name_lat_diff >= 0.05 or name_lon_diff >= 0.05:
                                        needs_disambiguation = True
                                        break
                        
                        # Also check for multiple distinct base locations (different names) if no coordinate-based disambiguation
                        if not needs_disambiguation:
                            # Extract base location names (before "/" or " / ") to identify distinct locations
                            base_locations = {}
                            for match in bus_stop_matches:
                                name = match.get('name', 'Unknown')
                                # Extract base location (before "/" or " / ")
                                base_name = name.split('/')[0].split(' / ')[0].strip()
                                if base_name not in base_locations:
                                    base_locations[base_name] = []
                                base_locations[base_name].append(name)
                            
                            # If there are multiple distinct base locations, disambiguate
                            if len(base_locations) > 1:
                                needs_disambiguation = True
                    
                    # If disambiguation needed, return error with enriched stop labels and options for reply handling.
                    # When Search API does not provide direction (e.g. Tudor Gardens x2), fetch Arrivals per stop
                    # to get platformName and towards so labels show "Stop AA (towards Willesden)".
                    if needs_disambiguation:
                        labelled_stops = []
                        disambiguation_options = []
                        for match in bus_stop_matches:
                            stop_id = match.get('id', '')
                            name = match.get('name') or stop_query
                            towards = (match.get('towards') or '').strip()
                            platform = self._platform_from_stop_id(stop_id)
                            # Enrich from Arrivals when Search didn't give direction (same logic as timetable grouping)
                            if (not towards or not platform) and stop_id:
                                arr_platform, arr_towards = self._get_stop_direction_from_arrivals(stop_id)
                                if arr_platform:
                                    platform = arr_platform
                                if arr_towards:
                                    towards = arr_towards
                            # Build label so user always sees direction when we have it (like bus_arrivals_grouped stop_label)
                            if platform and towards:
                                label = f"{name} – Stop {platform} (towards {towards})"
                            elif towards:
                                label = f"{name} (towards {towards})"
                            elif platform:
                                label = f"{name} – Stop {platform}"
                            else:
                                label = self._format_bus_stop_disambiguation_name(match, stop_query, enrich_from_arrivals=False)
                            labelled_stops.append(label)
                            disambiguation_options.append({
                                'id': stop_id,
                                'name': name,
                                'towards': towards,
                                'platform': platform,
                                'label': label
                            })
                        return {
                            'error': 'disambiguation_needed',
                            'query': stop_query,
                            'mode': 'bus',
                            'stations': labelled_stops,
                            'count': len(labelled_stops),
                            'disambiguation_options': disambiguation_options
                        }
                    # If all stops are at same location (coordinates within 0.05), proceed (show all stops together)
            
            for match in matches:
                stop_id = match.get('id')
                stop_name = match.get('name', stop_query)
                stop_modes = match.get('modes', [])
                
                if not stop_id or stop_id in stop_ids_processed:
                    continue
                
                # Determine if we should fetch this stop based on mode filter
                has_bus = 'bus' in stop_modes
                has_train = any(m in stop_modes for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])
                
                if mode_filter == 'bus' and not has_bus:
                    continue
                elif mode_filter == 'train' and not has_train:
                    continue
                elif mode_filter is None and not (has_bus or has_train):
                    continue
                
                stop_ids_processed.add(stop_id)
                stop_names.add(stop_name)
                
                # Get arrival predictions for this stop
                # Decide which StopPoint IDs to hit for Arrivals
                arrival_stop_ids = [stop_id]

                # For buses, expand group StopPoints (490G...) into leaf bus stops (4900...) :contentReference[oaicite:4]{index=4}
                if has_bus:
                    arrival_stop_ids = self._expand_to_leaf_bus_stops(stop_id)

                for arrival_id in arrival_stop_ids:
                    arrivals_url = f"{self.tfl_base_url}/StopPoint/{arrival_id}/Arrivals"
                    arrivals_params = self._tfl_params()

                    try:
                        arrivals_response = requests.get(arrivals_url, params=arrivals_params, timeout=10)
                        if arrivals_response.status_code == 200:
                            stop_arrivals = arrivals_response.json()
                            if stop_arrivals:
                                for arr in stop_arrivals:
                                    # Prefer stationName from prediction if present; otherwise use matched stop name
                                    arr["_stop_name"] = arr.get("stationName", stop_name)
                                    arr["_stop_id"] = arrival_id
                                    arr["_query_stop_id"] = stop_id  # original searched id (often the group)
                                    arr["_group_name"] = stop_name
                                all_arrivals.extend(stop_arrivals)
                    except Exception as e:
                        print(f"Error fetching arrivals for stop {arrival_id}: {e}")
                        continue
            
            if not all_arrivals:
                # Stop point was found but no arrivals available
                stop_name = list(stop_names)[0] if stop_names else stop_query
                
                # If mode_filter is 'train', check if any stops actually have train services
                # If no stops with train modes were found, return not_found instead
                if mode_filter == 'train':
                    has_train_stops = any(
                        any(m in match.get('modes', []) for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])
                        for match in matches
                    )
                    if not has_train_stops:
                        return {'error': 'not_found', 'query': stop_query, 'mode': 'train'}
                
                return {'error': 'no_arrivals', 'stop_name': stop_name, 'query': stop_query}
            
            # Use the most common stop name or original query
            if len(stop_names) == 1:
                stop_name = list(stop_names)[0]
            else:
                stop_name = stop_query
            
            first_stop_id = list(stop_ids_processed)[0] if stop_ids_processed else None
            result = self._build_timetable_from_arrivals(
                all_arrivals, stop_name, first_stop_id, mode_filter
            )
            if result is None:
                return None
            if mode_filter == 'train' and not result.get('train_arrivals'):
                has_train_stops = any(
                    any(m in match.get('modes', []) for m in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail'])
                    for match in matches
                    if match.get('id') in stop_ids_processed
                )
                if not has_train_stops:
                    return {'error': 'not_found', 'query': stop_query, 'mode': 'train', 'stop_name': stop_name}
            return result
        except Exception as e:
            print(f"Error fetching TFL timetable: {e}")
            return None

    def _build_timetable_from_arrivals(
        self,
        all_arrivals: List[Dict[str, Any]],
        stop_name: str,
        first_stop_id: Optional[str],
        mode_filter: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Build the timetable response dict from raw arrival list.
        Used by get_tfl_timetable and get_tfl_timetable_by_stop_id.
        """
        bus_arrivals: List[Dict[str, Any]] = []
        train_arrivals: List[Dict[str, Any]] = []

        def extract_direction(platform_name: str, destination: str) -> str:
            if not platform_name:
                return 'Unknown'
            platform_lower = platform_name.lower()
            if 'northbound' in platform_lower or 'north' in platform_lower:
                return 'Northbound'
            if 'southbound' in platform_lower or 'south' in platform_lower:
                return 'Southbound'
            if 'eastbound' in platform_lower or 'east' in platform_lower:
                return 'Eastbound'
            if 'westbound' in platform_lower or 'west' in platform_lower:
                return 'Westbound'
            if 'clockwise' in platform_lower:
                return 'Clockwise'
            if 'anticlockwise' in platform_lower or 'counter-clockwise' in platform_lower:
                return 'Anticlockwise'
            return 'Unknown'

        def extract_platform_number(platform_name: str) -> str:
            """Extract platform number from API platformName (e.g. 'Westbound - Platform 1' -> '1')."""
            if not platform_name:
                return ''
            m = re.search(r'platform\s*(\d+)', platform_name, re.IGNORECASE)
            return m.group(1) if m else ''

        for arr in all_arrivals:
            mode_name = arr.get('modeName', '').lower()
            time_to_station = arr.get('timeToStation', 0)
            minutes = round(time_to_station / 60, 1)
            platform_name = arr.get('platformName', '')
            destination = arr.get('destinationName', 'Unknown')
            stop_letter = (arr.get("platformName") or "").strip()
            towards = (arr.get("towards") or "").strip()
            if stop_letter and towards:
                stop_label = f"Stop {stop_letter} (towards {towards})"
            elif stop_letter:
                stop_label = f"Stop {stop_letter}"
            elif towards:
                stop_label = f"Towards {towards}"
            else:
                stop_label = (arr.get("_stop_name") or stop_name or "Stop").strip()
            direction = extract_direction(platform_name, destination)
            platform_number = extract_platform_number(platform_name) if mode_name in ('tube', 'train', 'dlr', 'overground', 'tram', 'national-rail') else ''
            arrival_info: Dict[str, Any] = {
                'line': arr.get('lineName', 'Unknown'),
                'destination': destination,
                'time_minutes': minutes,
                'time_seconds': time_to_station,
                'platform': platform_name,
                'direction': direction,
                'platform_number': platform_number,
                'vehicle_id': arr.get('vehicleId', ''),
                'stop_name': arr.get('_stop_name', stop_name),
                'group_id': arr.get('_query_stop_id') or arr.get('_stop_id'),
                'group_name': arr.get('_group_name', stop_name),
                'leaf_stop_id': arr.get('_stop_id'),
                'stop_label': stop_label
            }
            if mode_name in ['bus', 'coach']:
                bus_arrivals.append(arrival_info)
            elif mode_name in ['tube', 'train', 'dlr', 'overground', 'tram', 'national-rail']:
                train_arrivals.append(arrival_info)

        bus_arrivals.sort(key=lambda x: x['time_seconds'])
        train_arrivals.sort(key=lambda x: x['time_seconds'])

        if mode_filter == 'bus':
            train_arrivals = []
        elif mode_filter == 'train':
            bus_arrivals = []

        bus_arrivals_grouped: Dict[str, Any] = {}
        for b in bus_arrivals:
            gid = b.get('group_id') or 'unknown'
            gname = b.get('group_name') or stop_name
            stop_label = b.get('stop_label') or 'Stop'
            group_entry = bus_arrivals_grouped.setdefault(gid, {'group_name': gname, 'stops': {}})
            group_entry['stops'].setdefault(stop_label, []).append(b)
        for g in bus_arrivals_grouped.values():
            for arrs in g['stops'].values():
                arrs.sort(key=lambda x: x.get('time_seconds', 10**9))

        trains_by_direction: Dict[str, List[Dict[str, Any]]] = {}
        for train in train_arrivals:
            direction = train.get('direction', 'Unknown')
            trains_by_direction.setdefault(direction, []).append(train)
        buses_by_destination: Dict[str, List[Dict[str, Any]]] = {}
        for bus in bus_arrivals:
            destination = bus.get('destination', 'Unknown')
            buses_by_destination.setdefault(destination, []).append(bus)
        for destination_key in buses_by_destination:
            buses_by_destination[destination_key].sort(key=lambda x: x['time_seconds'])
        for direction in trains_by_direction:
            trains_by_direction[direction] = trains_by_direction[direction][:10]
        for destination_key in buses_by_destination:
            buses_by_destination[destination_key] = buses_by_destination[destination_key][:2]

        if not bus_arrivals and not train_arrivals:
            return None
        return {
            'stop_name': stop_name,
            'stop_id': first_stop_id,
            'bus_arrivals': bus_arrivals,
            'train_arrivals': train_arrivals,
            'bus_arrivals_by_destination': buses_by_destination,
            'train_arrivals_by_direction': trains_by_direction,
            'bus_arrivals_grouped': bus_arrivals_grouped,
            'timestamp': datetime.now().isoformat(),
            'source': 'TFL'
        }

    def get_tfl_timetable_by_stop_id(
        self,
        stop_id: str,
        mode_filter: Optional[str] = None,
        stop_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get timetable (arrivals) for a single TFL stop by ID.
        Used after direction disambiguation when the user has chosen one stop (e.g. 490000153AA).
        """
        if not self.has_tfl or not stop_id:
            return None
        mode_filter = mode_filter or 'bus'
        if not stop_name:
            try:
                info_url = f"{self.tfl_base_url}/StopPoint/{stop_id}"
                info_resp = requests.get(info_url, params=self._tfl_params(), timeout=5)
                if info_resp.status_code == 200:
                    info = info_resp.json()
                    stop_name = info.get('commonName') or info.get('name') or stop_id
                else:
                    stop_name = stop_id
            except Exception:
                stop_name = stop_id
        arrival_stop_ids = self._expand_to_leaf_bus_stops(stop_id)
        all_arrivals = []
        for aid in arrival_stop_ids:
            try:
                arrivals_url = f"{self.tfl_base_url}/StopPoint/{aid}/Arrivals"
                resp = requests.get(arrivals_url, params=self._tfl_params(), timeout=10)
                if resp.status_code == 200:
                    stop_arrivals = resp.json()
                    if stop_arrivals:
                        for arr in stop_arrivals:
                            arr["_stop_name"] = arr.get("stationName", stop_name)
                            arr["_stop_id"] = aid
                            arr["_query_stop_id"] = stop_id
                            arr["_group_name"] = stop_name
                        all_arrivals.extend(stop_arrivals)
            except Exception as e:
                print(f"Error fetching arrivals for stop {aid}: {e}")
        if not all_arrivals:
            return {'error': 'no_arrivals', 'stop_name': stop_name, 'query': stop_id}
        return self._build_timetable_from_arrivals(
            all_arrivals, stop_name, stop_id, mode_filter
        )

    def _platform_from_stop_id(self, stop_id: str) -> str:
        """
        Derive platform/stop letter from TFL stop id (e.g. 490000153AA -> AA, 490000153BB -> BB).
        """
        if not stop_id or len(stop_id) < 2:
            return ''
        # Common pattern: id ends with two letters for direction (AA, BB, etc.)
        tail = stop_id[-2:]
        if tail.isalpha():
            return tail.upper()
        return ''

    def _get_stop_direction_from_arrivals(self, stop_id: str) -> tuple:
        """
        Fetch Arrivals for a stop and return (platform, towards) from the first prediction.
        Used to enrich disambiguation labels when Search API does not provide direction info.
        Returns (platform_str, towards_str) - either may be empty.
        """
        if not stop_id or not self.has_tfl:
            return ('', '')
        try:
            arrival_ids = self._expand_to_leaf_bus_stops(stop_id)
            for aid in arrival_ids[:1]:  # first leaf only
                url = f"{self.tfl_base_url}/StopPoint/{aid}/Arrivals"
                resp = requests.get(url, params=self._tfl_params(), timeout=5)
                if resp.status_code == 200:
                    arrivals = resp.json()
                    if arrivals:
                        first = arrivals[0]
                        platform = (first.get('platformName') or '').strip()
                        towards = (first.get('towards') or '').strip()
                        return (platform, towards)
                break
        except Exception:
            pass
        return ('', '')

    def _format_bus_stop_disambiguation_name(
        self, stop_match: Dict[str, Any], fallback_query: str,
        enrich_from_arrivals: bool = False
    ) -> str:
        """
        Build a human-friendly label for a bus stop used in disambiguation prompts.
        Includes direction (towards) and optional platform (Stop AA/BB) when available.
        If enrich_from_arrivals is True and towards/platform are missing, fetches Arrivals
        to get platformName and towards (so e.g. "Tudor Gardens" becomes "Tudor Gardens – Stop AA (towards X)").
        """
        base_name = stop_match.get('name') or fallback_query or 'Unknown stop'
        towards = (stop_match.get('towards') or '').strip()
        stop_id = stop_match.get('id') or ''
        platform = self._platform_from_stop_id(stop_id)
        if (not towards or not platform) and enrich_from_arrivals and stop_id:
            arr_platform, arr_towards = self._get_stop_direction_from_arrivals(stop_id)
            if arr_platform:
                platform = arr_platform
            if arr_towards:
                towards = arr_towards
        if platform and towards:
            return f"{base_name} – Stop {platform} (towards {towards})"
        if towards:
            return f"{base_name} (towards {towards})"
        if platform:
            return f"{base_name} – Stop {platform}"
        return base_name

    def get_route_recommendation(self, origin: str, destination: str, avoid_tolls: bool = False, avoid_motorways: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get route recommendation between two points
        
        Args:
            origin: Starting location
            destination: Destination location
            
        Returns:
            dict: Route information including distance, duration, traffic level
        """
        # Try TFL Journey Planner first (for London)
        if self.has_tfl:
            try:
                tfl_journey = self._fetch_tfl_journey(origin, destination)
                if tfl_journey:
                    return tfl_journey
            except Exception as e:
                print(f"TFL Journey Planner error: {e}")
        
        # No API data available - return None instead of mock data
        return None
    
    
    # Map display names (from _train_lines) to TfL Line/Mode API ids
    _TRAIN_LINE_DISPLAY_TO_ID = {
        'bakerloo': 'bakerloo',
        'central': 'central',
        'circle': 'circle',
        'district': 'district',
        'hammersmith & city': 'hammersmith-city',
        'hammersmith and city': 'hammersmith-city',
        'jubilee': 'jubilee',
        'metropolitan': 'metropolitan',
        'northern': 'northern',
        'piccadilly': 'piccadilly',
        'victoria': 'victoria',
        'waterloo & city': 'waterloo-city',
        'waterloo and city': 'waterloo-city',
        'london overground': None,  # no single id; user must name a specific line
        'windrush': 'windrush',
        'lioness': 'lioness',
        'mildmay': 'mildmay',
        'suffragette': 'suffragette',
        'weaver': 'weaver',
        'liberty': 'liberty',
        'dlr': 'dlr',
        'docklands light railway': 'dlr',
    }

    def get_transit_disruption(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Get transit disruption for a train line (Underground, Overground, DLR).
        Uses Line/Mode/tube,dlr,overground/Status; returns status or error dict.
        Returns: dict with status/description, or dict with 'error': 'line_not_found'|'api_error'.
        """
        if not self.has_tfl:
            return {'error': 'api_error'}
        line_normalized = line.lower().strip()
        line_normalized = re.sub(r"['\u2019]", '', line_normalized)
        line_normalized = re.sub(r'\s*&\s*', ' and ', line_normalized)
        line_normalized = re.sub(r'\s+', ' ', line_normalized).strip()
        # Also try with " and " -> " & " for key lookup
        line_id = self._TRAIN_LINE_DISPLAY_TO_ID.get(line_normalized)
        if line_id is None and ' and ' in line_normalized:
            line_id = self._TRAIN_LINE_DISPLAY_TO_ID.get(line_normalized.replace(' and ', ' & '))
        if line_id is None:
            # Allow "windrush line" -> windrush
            for key, val in self._TRAIN_LINE_DISPLAY_TO_ID.items():
                if val and (key in line_normalized or (key + ' line') in line_normalized):
                    line_id = val
                    break
        if not line_id:
            return {'error': 'line_not_found', 'line': line}
        try:
            url = f"{self.tfl_base_url}/Line/Mode/tube,dlr,overground/Status"
            params = {**self._tfl_params(), 'detail': True}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching transit disruption: {e}")
            return {'error': 'api_error'}
        except Exception as e:
            print(f"Error parsing transit disruption response: {e}")
            return {'error': 'api_error'}
        if not isinstance(data, list):
            return {'error': 'api_error'}
        line_obj = None
        for item in data:
            if isinstance(item, dict) and item.get('id') == line_id:
                line_obj = item
                break
        if not line_obj:
            return {'error': 'line_not_found', 'line': line}
        line_name = line_obj.get('name', line)
        disruptions = []
        detailed_descriptions = []
        affected_locations = []
        status = 'Good Service'
        for status_item in line_obj.get('lineStatuses', []):
            status_text = (status_item.get('statusSeverityDescription') or '').strip()
            if status_text and status_text != 'Good Service':
                disruptions.append(status_text)
                if status == 'Good Service':
                    status = status_text
                reason = (status_item.get('reason') or '').strip()
                if reason:
                    detailed_descriptions.append(reason)
                for period in status_item.get('validityPeriods', []):
                    from_s = period.get('fromStation')
                    to_s = period.get('toStation')
                    if from_s and to_s:
                        from_name = from_s.get('commonName', '') if isinstance(from_s, dict) else str(from_s)
                        to_name = to_s.get('commonName', '') if isinstance(to_s, dict) else str(to_s)
                        if from_name and to_name:
                            loc = f"between {from_name} and {to_name}"
                            if loc not in affected_locations:
                                affected_locations.append(loc)
        if not detailed_descriptions and disruptions:
            detailed_descriptions = list(disruptions)
        description = '; '.join(detailed_descriptions) if detailed_descriptions else status
        line_variations = [f"{line_name} Line:", f"{line_name}:", f"{line_name}Line:", line_name]
        for var in line_variations:
            if description.startswith(var):
                description = description[len(var):].strip().lstrip(': -')
                break
        return {
            'line': line_name,
            'status': status,
            'description': description or status,
            'affected_locations': affected_locations,
            'alternatives': []
        }
    
    def get_bus_disruption(self, route: str) -> Optional[Dict[str, Any]]:
        """Get bus disruption for a specific bus route"""
        if self.has_tfl:
            try:
                # Extract bus route number from query (e.g., "83", "302", "N83")
                bus_route_pattern = r'\b([Nn]?\d{1,3})\b'
                route_matches = re.findall(bus_route_pattern, route)
                if not route_matches:
                    # If no route number found, try using the route string directly
                    route_id = route.strip()
                else:
                    # Use the first route number found
                    route_id = route_matches[0].upper()
                
                # Fetch bus route status from TFL API
                url = f"{self.tfl_base_url}/Line/{route_id}/Status"
                params = {
                    'app_id': self.tfl_app_id,
                    'app_key': self.tfl_app_key,
                    'detail': True
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        line_status = data[0]
                        line_name = line_status.get('name', route_id)
                        
                        # Check for disruptions
                        disruptions = []
                        detailed_descriptions = []
                        affected_locations = []
                        delay_minutes = 0
                        status = 'Good Service'
                        
                        # Process lineStatuses
                        for status_item in line_status.get('lineStatuses', []):
                            status_text = status_item.get('statusSeverityDescription', '')
                            if status_text and status_text != 'Good Service':
                                disruptions.append(status_text)
                                
                                # Extract detailed reason if available
                                reason = status_item.get('reason', '')
                                if reason:
                                    detailed_descriptions.append(reason)
                                
                                # Extract affected locations from validityPeriods
                                validity_periods = status_item.get('validityPeriods', [])
                                for period in validity_periods:
                                    from_station = period.get('fromStation')
                                    to_station = period.get('toStation')
                                    
                                    if from_station and to_station:
                                        from_name = from_station.get('commonName', '') if isinstance(from_station, dict) else str(from_station)
                                        to_name = to_station.get('commonName', '') if isinstance(to_station, dict) else str(to_station)
                                        if from_name and to_name:
                                            location_str = f"between {from_name} and {to_name}"
                                            if location_str not in affected_locations:
                                                affected_locations.append(location_str)
                                
                                # Estimate delay based on severity
                                if 'Severe' in status_text or 'Closed' in status_text:
                                    delay_minutes = max(delay_minutes, 30)
                                elif 'Minor' in status_text:
                                    delay_minutes = max(delay_minutes, 5)
                                else:
                                    delay_minutes = max(delay_minutes, 15)
                                status = status_text
                        
                        # Fallback to disruptions list if lineStatuses is empty
                        if not disruptions:
                            for disruption in line_status.get('disruptions', []):
                                desc = (
                                    disruption.get('description')
                                    or disruption.get('categoryDescription')
                                    or disruption.get('category')
                                )
                                if desc:
                                    disruptions.append(desc)
                                    if status == 'Good Service':
                                        status = desc
                                    
                                    # Extract affected locations from disruption periods
                                    disruption_periods = disruption.get('validityPeriods', [])
                                    for period in disruption_periods:
                                        from_station = period.get('fromStation')
                                        to_station = period.get('toStation')
                                        
                                        if from_station and to_station:
                                            from_name = from_station.get('commonName', '') if isinstance(from_station, dict) else str(from_station)
                                            to_name = to_station.get('commonName', '') if isinstance(to_station, dict) else str(to_station)
                                            if from_name and to_name:
                                                location_str = f"between {from_name} and {to_name}"
                                                if location_str not in affected_locations:
                                                    affected_locations.append(location_str)
                                    
                                    # Estimate delay based on severity
                                    severity = (disruption.get('severity') or '').lower()
                                    if 'severe' in severity or 'closure' in severity or 'closed' in severity:
                                        delay_minutes = max(delay_minutes, 30)
                                    elif 'minor' in severity or 'reduced' in severity:
                                        delay_minutes = max(delay_minutes, 5)
                                    else:
                                        delay_minutes = max(delay_minutes, 15)
                        
                        # Only return if there's an actual disruption
                        if status != 'Good Service' or delay_minutes > 0:
                            # Prefer detailed descriptions over generic status text
                            if detailed_descriptions:
                                description = '; '.join(detailed_descriptions)
                            elif disruptions:
                                # Use disruptions list, but avoid duplicating status
                                disruption_texts = [d for d in disruptions if d != status]
                                if disruption_texts:
                                    description = '; '.join(disruption_texts)
                                else:
                                    description = status
                            else:
                                description = status
                            
                            # Remove route name from description if it starts with it
                            route_variations = [
                                f"Bus {route_id}:",
                                f"Route {route_id}:",
                                f"{route_id}:",
                                f"Bus {route_id}",
                                f"Route {route_id}",
                                route_id
                            ]
                            description_cleaned = description
                            for route_var in route_variations:
                                if description_cleaned.startswith(route_var):
                                    description_cleaned = description_cleaned[len(route_var):].strip()
                                    description_cleaned = description_cleaned.lstrip(': -')
                                    break
                            
                            return {
                                'route': line_name,
                                'status': status,
                                'description': description_cleaned,
                                'affected_locations': affected_locations,
                                'delay_minutes': delay_minutes,
                                'alternatives': []
                            }
            except Exception as e:
                print(f"Error fetching bus disruption: {e}")
        
        return None
    
    def get_transit_route(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """Get public transport route between origin and destination"""
        if self.has_tfl:
            try:
                journey = self._fetch_tfl_journey(origin, destination)
                if journey:
                    return journey
            except Exception as e:
                print(f"Error fetching transit route: {e}")
        
        # No API data available - return None instead of mock data
        return None
    
    
    def _fetch_tfl_line_status(self, route: str) -> Optional[Dict[str, Any]]:
        """
        Fetch line status from TFL API
        Supports both specific line queries (/Line/{line_id}/Status) 
        and mode-based queries (/Line/mode/{mode}/status)
        Based on: https://api.tfl.gov.uk/line/mode/tube/status
        """
        # Map common route names to TFL line IDs
        line_mapping = {
            'bakerloo': 'bakerloo',
            'central': 'central',
            'circle': 'circle',
            'district': 'district',
            'hammersmith': 'hammersmith-city',
            'hammersmith-city': 'hammersmith-city',
            'jubilee': 'jubilee',
            'metropolitan': 'metropolitan',
            'northern': 'northern',
            'piccadilly': 'piccadilly',
            'victoria': 'victoria',
            'waterloo': 'waterloo-city',
            'waterloo-city': 'waterloo-city',
            'dlr': 'dlr',
            'overground': 'london-overground',
            'london-overground': 'london-overground',
            'tram': 'tram',
            'bus': 'bus'
        }
        
        route_lower = route.lower().strip()
        line_id = None
        
        # Try to find matching line (exact match first)
        for key, value in line_mapping.items():
            if key in route_lower:
                line_id = value
                break
        
        # If no exact match, try the route name directly (formatted)
        if not line_id:
            line_id = route_lower.replace(' ', '-').replace('_', '-')
        
        try:
            # Use specific line status endpoint: /Line/{line_id}/Status
            url = f"{self.tfl_base_url}/Line/{line_id}/Status"
            params = {
                'app_id': self.tfl_app_id,
                'app_key': self.tfl_app_key,
                'detail': True
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                line_status = data[0]
                line_name = line_status.get('name', route)
                
                # Check for disruptions
                disruptions = []
                detailed_descriptions = []  # More detailed descriptions from reason field
                affected_locations = []  # Location/segment information
                delay_minutes = 0
                status = 'Good Service'
                
                # 1) Use lineStatuses from the /Status endpoint when available
                has_validity_periods = False
                for status_item in line_status.get('lineStatuses', []):
                    status_text = status_item.get('statusSeverityDescription', '')
                    if status_text and status_text != 'Good Service':
                        disruptions.append(status_text)
                        # Extract detailed reason if available (more specific than status)
                        reason = status_item.get('reason', '')
                        if reason and reason.strip():
                            detailed_descriptions.append(reason.strip())
                        
                        # Extract location/segment information from validityPeriods
                        status_validity_periods = status_item.get('validityPeriods', [])
                        if status_validity_periods:
                            has_validity_periods = True
                            
                        for period in status_validity_periods:
                            from_station = period.get('fromStation')
                            to_station = period.get('toStation')
                            
                            if from_station and to_station:
                                # Specific segment
                                from_name = from_station.get('commonName', '') if isinstance(from_station, dict) else str(from_station)
                                to_name = to_station.get('commonName', '') if isinstance(to_station, dict) else str(to_station)
                                if from_name and to_name:
                                    location_str = f"between {from_name} and {to_name}"
                                    if location_str not in affected_locations:
                                        affected_locations.append(location_str)
                            elif from_station:
                                # Single station or starting point
                                from_name = from_station.get('commonName', '') if isinstance(from_station, dict) else str(from_station)
                                if from_name:
                                    location_str = f"from {from_name}"
                                    if location_str not in affected_locations:
                                        affected_locations.append(location_str)
                        
                        # Estimate delay based on severity text
                        if 'Severe' in status_text or 'Closed' in status_text:
                            delay_minutes = max(delay_minutes, 30)
                        elif 'Minor' in status_text:
                            delay_minutes = max(delay_minutes, 5)
                        else:
                            delay_minutes = max(delay_minutes, 15)
                        status = status_text
                
                # 2) Fallback: use top-level `disruptions` list when present
                # This matches the shape shown in the sample TFL response the user provided,
                # where each line object has a `disruptions` array alongside `lineStatuses`.
                if not disruptions:
                    for disruption in line_status.get('disruptions', []):
                        # Prefer a human‑readable description if available
                        desc = (
                            disruption.get('description')
                            or disruption.get('categoryDescription')
                            or disruption.get('category')
                        )
                        if not desc:
                            continue
                        disruptions.append(desc)
                        # Also extract detailed description if available
                        detailed_desc = disruption.get('description', '')
                        if detailed_desc and detailed_desc.strip() and detailed_desc not in detailed_descriptions:
                            detailed_descriptions.append(detailed_desc.strip())
                        
                        # Extract location information from disruption if available
                        affected_routes = disruption.get('affectedRoutes', [])
                        affected_stops = disruption.get('affectedStops', [])
                        
                        # Check validityPeriods in disruption
                        disruption_periods = disruption.get('validityPeriods', [])
                        for period in disruption_periods:
                            from_station = period.get('fromStation')
                            to_station = period.get('toStation')
                            
                            if from_station and to_station:
                                from_name = from_station.get('commonName', '') if isinstance(from_station, dict) else str(from_station)
                                to_name = to_station.get('commonName', '') if isinstance(to_station, dict) else str(to_station)
                                if from_name and to_name:
                                    location_str = f"between {from_name} and {to_name}"
                                    if location_str not in affected_locations:
                                        affected_locations.append(location_str)
                        
                        # Severity-based rough delay estimation if a severity field is present
                        severity = (disruption.get('severity') or '').lower()
                        if 'severe' in severity or 'closure' in severity or 'closed' in severity:
                            delay_minutes = max(delay_minutes, 30)
                        elif 'minor' in severity or 'reduced' in severity:
                            delay_minutes = max(delay_minutes, 5)
                        else:
                            # Generic disruption with unknown severity – treat as moderate
                            delay_minutes = max(delay_minutes, 15)
                        if status == 'Good Service':
                            status = desc
                
                # Only calculate derived values if we have actual status / disruption data
                # Don't add default/synthetic congestion or speed values
                result = {
                    'route': line_name,
                    'status': status,
                    'delay_minutes': delay_minutes,
                    'disruptions': disruptions,
                    'detailed_descriptions': detailed_descriptions,  # More specific disruption details
                    'affected_locations': affected_locations,  # Location/segment information
                    'timestamp': datetime.now().isoformat(),
                    'source': 'TFL'
                }
                
                # Only add congestion_level if we have a non‑zero delay (i.e. some disruption)
                if delay_minutes > 0:
                    congestion_level = 'light' if delay_minutes < 10 else 'moderate' if delay_minutes < 20 else 'heavy'
                    result['congestion_level'] = congestion_level
                
                return result
        except requests.exceptions.RequestException as e:
            print(f"TFL API request failed: {e}")
            return None
        except Exception as e:
            print(f"Error parsing TFL response: {e}")
            return None
    
    def _fetch_tfl_journey(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Fetch journey planning from TFL API
        Supports: station names, lat/lon (51.501,-0.123), postcodes (n225nb), stop IDs
        Based on: https://api.tfl.gov.uk/journey/journeyresults/{origin}/to/{destination}
        """
        try:
            # URL encode origin and destination to handle spaces and special characters
            # TFL API supports: station names, lat/lon (51.501,-0.123), postcodes, stop IDs
            origin_encoded = quote(origin, safe='')
            destination_encoded = quote(destination, safe='')
            
            url = f"{self.tfl_base_url}/Journey/JourneyResults/{origin_encoded}/to/{destination_encoded}"
            params = {
                'app_id': self.tfl_app_id,
                'app_key': self.tfl_app_key,
                'mode': 'tube,bus,dlr,overground,tram,walking',
                'journeyPreference': 'leastTime'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Handle disambiguation - API may return multiple options for start/end points
            # Check if we need to disambiguate (fromJourney and toJourney contain multiple options)
            if 'fromLocationDisambiguation' in data or 'toLocationDisambiguation' in data:
                # If disambiguation needed, use first option or return disambiguation info
                if 'fromLocationDisambiguation' in data:
                    disambiguation_options = data['fromLocationDisambiguation'].get('disambiguationOptions', [])
                    if disambiguation_options:
                        # Use first option's identifier
                        origin_encoded = quote(disambiguation_options[0].get('parameterValue', origin), safe='')
                        url = f"{self.tfl_base_url}/Journey/JourneyResults/{origin_encoded}/to/{destination_encoded}"
                        response = requests.get(url, params=params, timeout=10)
                        response.raise_for_status()
                        data = response.json()
                
                if 'toLocationDisambiguation' in data:
                    disambiguation_options = data['toLocationDisambiguation'].get('disambiguationOptions', [])
                    if disambiguation_options:
                        destination_encoded = quote(disambiguation_options[0].get('parameterValue', destination), safe='')
                        url = f"{self.tfl_base_url}/Journey/JourneyResults/{origin_encoded}/to/{destination_encoded}"
                        response = requests.get(url, params=params, timeout=10)
                        response.raise_for_status()
                        data = response.json()
            
            if 'journeys' in data and len(data['journeys']) > 0:
                journey = data['journeys'][0]
                duration_seconds = journey.get('duration', 0)
                duration_minutes = round(duration_seconds / 60)
                
                # Check for disruptions
                disruptions = []
                for leg in journey.get('legs', []):
                    disruptions.extend(leg.get('disruptions', []))
                
                traffic_level = 'heavy' if disruptions else 'moderate'
                
                # Try to get actual distance from journey data
                distance_meters = 0
                for leg in journey.get('legs', []):
                    distance_meters += leg.get('distance', 0)
                
                if distance_meters > 0:
                    distance_km = round(distance_meters / 1000, 1)
                else:
                    # Estimate distance if not provided
                    distance_km = round((duration_minutes * 0.5), 1)  # Rough estimate: 30 km/h average
                
                return {
                    'origin': origin,
                    'destination': destination,
                    'distance_km': distance_km,
                    'duration_minutes': duration_minutes,
                    'traffic_level': traffic_level,
                    'disruptions': len(disruptions) > 0,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'TFL'
                }
        except Exception as e:
            print(f"Error fetching TFL journey: {e}")
            return None
    
    
    