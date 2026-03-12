"""
Transport Data Fetcher
Integrates with real-world transport APIs (TfL, Google Maps).
Timetables, line status, disruptions, route recommendations, stop info.
Returns None or empty list if no API data is available
"""
import requests
import os
import re
import math
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from urllib.parse import quote


class TransportDataFetcher:
    def __init__(self):
        """Initialize transport API clients"""
        self.tfl_app_id = os.environ.get('TFL_APP_ID', '')
        self.tfl_app_key = os.environ.get('TFL_APP_KEY', '')
        self.tfl_base_url = 'https://api.tfl.gov.uk'

        self.google_api_key = os.environ.get('GOOGLE_MAPS_API_KEY', '')
        self.google_maps_base_url = 'https://maps.googleapis.com/maps/api'

        self.has_tfl = bool(self.tfl_app_id and self.tfl_app_key)
        self.has_google = bool(self.google_api_key)

        self._TRAIN_LINE_DISPLAY_TO_ID = {
            'bakerloo': 'bakerloo',
            'central': 'central',
            'circle': 'circle',
            'district': 'district',
            'hammersmith-city': 'hammersmith-city',
            'jubilee': 'jubilee',
            'metropolitan': 'metropolitan',
            'northern': 'northern',
            'piccadilly': 'piccadilly',
            'victoria': 'victoria',
            'waterloo-city': 'waterloo-city',
            'windrush': 'windrush',
            'lioness': 'lioness',
            'mildmay': 'mildmay',
            'suffragette': 'suffragette',
            'weaver': 'weaver',
            'liberty': 'liberty',
            'dlr': 'dlr',
        }

        if self.has_tfl:
            print("TFL API initialized (App ID and Key configured)")
        if self.has_google:
            print("Google Maps API initialized")

    def _tfl_params(self) -> Dict[str, str]:
        """Common TfL auth params"""
        params = {}
        if self.tfl_app_id:
            params["app_id"] = self.tfl_app_id
        if self.tfl_app_key:
            params["app_key"] = self.tfl_app_key
        return params

    def _normalize_tube_station_name(self, name: str) -> str:
        """Normalize station name with word-by-word title casing"""
        if not name:
            return name

        small_words = {'a', 'an', 'and', 'or', 'the', 'of', 'in', 'on', 'at', 'to', 'for'}
        words = name.split()
        normalized = []

        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                if '-' in word:
                    parts = word.split('-')
                    word = '-'.join(p.title() for p in parts)
                elif "'" in word:
                    parts = word.split("'")
                    word = "'".join(p.title() for p in parts)
                else:
                    word = word.title()
            else:
                word = word.lower()

            if word == 'St':
                word = 'St'

            normalized.append(word)

        return ' '.join(normalized)

    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        if word1 == word2:
            return 1.0

        set1 = set(word1.lower())
        set2 = set(word2.lower())

        if not set1 or not set2:
            return 0.0

        overlap = len(set1 & set2)
        union = len(set1 | set2)
        char_overlap = overlap / union if union > 0 else 0.0

        first_bonus = 0.1 if word1[0].lower() == word2[0].lower() else 0.0
        last_bonus = 0.1 if word1[-1].lower() == word2[-1].lower() else 0.0

        len_diff = abs(len(word1) - len(word2))
        len_similarity = 1.0 - (len_diff / max(len(word1), len(word2)))

        return (char_overlap + first_bonus + last_bonus + len_similarity * 0.2) / 2.3

    def _find_closest_location_match(self, query: str, options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the closest matching location from options using multiple scoring methods"""
        if not options or not query:
            return None

        query_lower = query.lower().strip()
        best_option = None
        best_score = 0.0

        for option in options:
            name = option.get('name', '').lower()
            if not name:
                continue

            if name == query_lower:
                return option

            cleaned_name = re.sub(r'[^a-z0-9\s]', '', name)
            cleaned_query = re.sub(r'[^a-z0-9\s]', '', query_lower)

            scores = []

            if cleaned_name == cleaned_query:
                scores.append(0.95)

            if query_lower in name:
                scores.append(0.85)

            if name in query_lower:
                scores.append(0.80)

            query_words = query_lower.split()
            name_words = name.split()
            word_matches = sum(1 for qw in query_words if any(qw in nw for nw in name_words))
            if query_words:
                scores.append(word_matches / len(query_words) * 0.75)

            char_overlap = sum(1 for c in query_lower if c in name) / len(query_lower) if query_lower else 0
            scores.append(char_overlap * 0.60)

            substring_bonus = 0.70 if any(qw in name for qw in query_words) else 0.0
            scores.append(substring_bonus)

            if query_words and name_words:
                first_word_match = 1.0 if query_words[0] in name_words[0] else 0.0
                scores.append(first_word_match * 0.65)

            avg_score = sum(scores) / len(scores) if scores else 0.0

            if avg_score > best_score:
                best_score = avg_score
                best_option = option

        return best_option if best_score > 0.4 else None

    def _expand_to_leaf_bus_stops(self, stop_id: str) -> List[str]:
        """
        If stop_id is a StopPoint group/area, expand to leaf bus stop IDs.
        Returns just the stop_id if not expandable or expansion not available.
        """
        if not stop_id:
            return []

        if not self.has_tfl:
            return [stop_id]

        try:
            url = f"{self.tfl_base_url}/StopPoint/{stop_id}"
            resp = requests.get(url, params=self._tfl_params(), timeout=5)
            resp.raise_for_status()
            data = resp.json()

            children = data.get("children", [])
            if children:
                child_ids = [c.get("id") for c in children if c.get("id")]
                return child_ids if child_ids else [stop_id]
        except Exception:
            pass

        return [stop_id]

    def _platform_from_stop_id(self, stop_id: str) -> Optional[str]:
        """Get platform information for a stop."""
        return None

    def _get_stop_direction_from_arrivals(self, stop_id: str, mode: str = 'bus') -> Optional[str]:
        """Get direction/destination from timetable arrivals."""
        return None

    def _format_bus_stop_disambiguation_name(self, stop_id: str, platform: Optional[str] = None, direction: Optional[str] = None) -> str:
        """Format a stop name with platform/direction for disambiguation."""
        name_parts = [stop_id]
        if platform:
            name_parts.append(f"Stop {platform}")
        if direction:
            name_parts.append(f"towards {direction}")
        return ' - '.join(name_parts)

    def get_route_traffic(self, route: str) -> Optional[Dict[str, Any]]:
        """
        Get traffic information for a specific route

        Args:
            route: Route identifier (e.g., "Bakerloo", "Central")

        Returns:
            dict: Traffic data including status and delay
        """
        if self.has_tfl:
            try:
                tfl_data = self._fetch_tfl_line_status(route)
                if tfl_data:
                    return tfl_data
            except Exception as e:
                print(f"TFL API error: {e}")

        return None

    def _fetch_tfl_line_status(self, line_name: str) -> Optional[Dict[str, Any]]:
        """Fetch TfL line status"""
        if not self.has_tfl:
            return None

        try:
            url = f"{self.tfl_base_url}/Line/{quote(line_name)}/Status"
            resp = requests.get(url, params=self._tfl_params(), timeout=5)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and len(data) > 0:
                line_data = data[0]
                statuses = line_data.get('lineStatuses', [])
                if statuses:
                    status = statuses[0]
                    return {
                        'line': line_name,
                        'status': status.get('statusSeverityDescription', 'Unknown'),
                        'description': status.get('reason', ''),
                    }
        except Exception as e:
            print(f"TFL fetch error: {e}")

        return None

    def get_transit_disruption(self, line_name: str, mode: str = 'tube') -> Optional[Dict[str, Any]]:
        """
        Get disruption information for a transit line

        Args:
            line_name: Line name (e.g., "Central", "Northern")
            mode: 'tube', 'dlr', or 'overground'

        Returns:
            dict: Disruption data with status and affected locations
        """
        if not self.has_tfl:
            return None

        try:
            url = f"{self.tfl_base_url}/Line/{quote(line_name)}/Status"
            params = self._tfl_params()
            params['detail'] = 'true'

            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and len(data) > 0:
                line_data = data[0]
                statuses = line_data.get('lineStatuses', [])
                if statuses:
                    status = statuses[0]
                    disruption_reason = status.get('reason', '')

                    line_name_clean = disruption_reason.replace(' line', '').replace(' status', '')

                    affected_locations = []
                    validity_periods = status.get('validityPeriods', [])
                    for period in validity_periods:
                        if 'fromDate' in period:
                            affected_locations.append(period['fromDate'])

                    return {
                        'line': line_name,
                        'status': status.get('statusSeverityDescription', 'Unknown'),
                        'description': disruption_reason,
                        'affected_locations': affected_locations,
                    }
        except Exception as e:
            print(f"TFL disruption error: {e}")

        return None

    def get_bus_disruption(self, route: str) -> Optional[Dict[str, Any]]:
        """
        Get disruption information for a bus route

        Args:
            route: Bus route number

        Returns:
            dict: Disruption data with status and details
        """
        if not self.has_tfl:
            return None

        try:
            url = f"{self.tfl_base_url}/Line/{quote(route)}/Status"
            params = self._tfl_params()
            params['detail'] = 'true'

            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and len(data) > 0:
                line_data = data[0]
                statuses = line_data.get('lineStatuses', [])
                if statuses:
                    status = statuses[0]

                    route_match = re.search(r'\b(\d{1,3})\b', status.get('reason', ''))
                    extracted_route = route_match.group(1) if route_match else route

                    return {
                        'route': extracted_route,
                        'status': status.get('statusSeverityDescription', 'Unknown'),
                        'description': status.get('reason', ''),
                        'severity': status.get('statusSeverity', 'Unknown'),
                    }
        except Exception as e:
            print(f"TFL bus disruption error: {e}")

        return None

    def get_tfl_timetable(self, query: str, mode_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get timetable for a location with multiple retry strategies

        Args:
            query: Location query
            mode_filter: Optional filter ('bus', 'train', etc.)

        Returns:
            dict: Structured timetable with bus_arrivals, train_arrivals, etc.
        """
        if not self.has_tfl:
            return None

        queries_to_try = [query]
        if ' ' in query:
            queries_to_try.append(query.split()[0])
            for word in query.split():
                queries_to_try.append(word)

        for q in queries_to_try:
            try:
                url = f"{self.tfl_base_url}/StopPoint/Search"
                params = self._tfl_params()
                params['query'] = q
                params['modes'] = mode_filter or 'bus,tube,overground,dlr'

                resp = requests.get(url, params=params, timeout=5)
                resp.raise_for_status()
                data = resp.json()

                matches = data.get('matches', [])
                if matches:
                    stop = matches[0]
                    stop_id = stop.get('id')

                    return self._build_timetable_from_arrivals(stop_id, stop.get('name', query))
            except Exception:
                pass

        return None

    def _build_timetable_from_arrivals(self, stop_id: str, stop_name: str) -> Optional[Dict[str, Any]]:
        """Build timetable dict from stop arrivals data"""
        try:
            url = f"{self.tfl_base_url}/StopPoint/{stop_id}/Arrivals"
            resp = requests.get(url, params=self._tfl_params(), timeout=5)
            resp.raise_for_status()
            data = resp.json()

            bus_arrivals = []
            train_arrivals = []
            bus_arrivals_grouped = {}
            train_arrivals_by_direction = {}
            bus_arrivals_by_destination = {}

            if isinstance(data, list):
                for arrival in data[:20]:
                    line_id = arrival.get('lineId', '')
                    line_name = arrival.get('lineName', '')
                    mode = arrival.get('vehicleId', '').split('.')[0] if 'vehicleId' in arrival else ''
                    destination = arrival.get('destinationName', '')
                    time_to_station = arrival.get('timeToStation', 0)
                    expected_arrival = arrival.get('expectedArrival', '')
                    group_id = arrival.get('naptanId', '')
                    stop_label = arrival.get('platformName', '')

                    arrival_dict = {
                        'lineId': line_id,
                        'lineName': line_name,
                        'destination': destination,
                        'timeToStation': time_to_station,
                        'expectedArrival': expected_arrival,
                    }

                    if 'bus' in mode.lower() or line_id.startswith('1'):
                        bus_arrivals.append(arrival_dict)
                        if group_id not in bus_arrivals_grouped:
                            bus_arrivals_grouped[group_id] = []
                        bus_arrivals_grouped[group_id].append(arrival_dict)
                        if destination not in bus_arrivals_by_destination:
                            bus_arrivals_by_destination[destination] = []
                        bus_arrivals_by_destination[destination].append(arrival_dict)
                    else:
                        train_arrivals.append(arrival_dict)
                        if destination not in train_arrivals_by_direction:
                            train_arrivals_by_direction[destination] = []
                        train_arrivals_by_direction[destination].append(arrival_dict)

            return {
                'stop_name': stop_name,
                'stop_id': stop_id,
                'bus_arrivals': bus_arrivals,
                'train_arrivals': train_arrivals,
                'bus_arrivals_grouped': bus_arrivals_grouped,
                'train_arrivals_by_direction': train_arrivals_by_direction,
                'bus_arrivals_by_destination': bus_arrivals_by_destination,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'TfL API',
            }
        except Exception:
            pass

        return None

    def get_tfl_timetable_by_stop_id(self, stop_id: str, mode: str = 'bus') -> Optional[List[Dict[str, Any]]]:
        """
        Get timetable for a specific stop by ID.

        Args:
            stop_id: TfL stop ID
            mode: 'bus' or 'train'

        Returns:
            list: Timetable entries with arrival times
        """
        if not self.has_tfl:
            return None

        try:
            url = f"{self.tfl_base_url}/StopPoint/{stop_id}/Arrivals"
            resp = requests.get(url, params=self._tfl_params(), timeout=5)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list):
                arrivals = []
                for arrival in data[:5]:
                    arrivals.append({
                        'lineId': arrival.get('lineId'),
                        'lineName': arrival.get('lineName'),
                        'destination': arrival.get('destinationName'),
                        'timeToStation': arrival.get('timeToStation'),
                        'expectedArrival': arrival.get('expectedArrival'),
                    })
                return arrivals if arrivals else None
        except Exception:
            pass

        return None

    def get_route_recommendation(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """Get route recommendation stub"""
        return None

    def get_stop_timetable(self, stop_id: str, mode: str = 'bus') -> Optional[List[Dict[str, Any]]]:
        """
        Get timetable for a specific stop

        Args:
            stop_id: Stop identifier
            mode: 'bus' or 'train'

        Returns:
            list: Timetable entries
        """
        if not self.has_tfl:
            return None

        return None
