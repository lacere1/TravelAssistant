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
            print("Warning: No API keys found.")
    
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

        In this earlier-stage version we keep things simple and just return
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
        
        # No API data available
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
        if not station_name or not station_name.strip():
            return station_name
        return ' '.join(w.capitalize() for w in station_name.strip().split())
    
    def _find_closest_location_match(self, location: str, all_stops: List[Dict[str, Any]]) -> Optional[str]:
        if not all_stops:
            return None
        location_lower = location.lower()
        best_match = None
        best_score = 0.0
        for stop in all_stops:
            name = stop.get('name', '')
            if not name:
                continue
            score = SequenceMatcher(None, location_lower, name.lower()).ratio()
            if location_lower in name.lower() or name.lower() in location_lower:
                score = max(score, 0.75)
            if score > best_score:
                best_score = score
                best_match = name
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
            
            params = {
                'app_id': self.tfl_app_id,
                'app_key': self.tfl_app_key,
                'query': search_query,
                'modes': 'tube,bus,dlr,overground,tram'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check if any stop points were found
            matches = data.get('matches', [])
            
            if not matches:
                # Retry with cleaned query
                common_words = ['station', 'stop', 'the', 'tube', 'underground', 'metro']
                cleaned_query = ' '.join(w for w in stop_query.split() if w.lower() not in common_words).strip()
                if cleaned_query and cleaned_query.lower() != stop_query.lower():
                    params['query'] = cleaned_query
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        matches = response.json().get('matches', [])

                # Broader: first main word
                if not matches:
                    main_words = [w for w in stop_query.split() if len(w) > 2 and w.lower() not in common_words]
                    if main_words:
                        params['query'] = main_words[0]
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 200:
                            broader = response.json().get('matches', [])
                            if broader:
                                closest = self._find_closest_location_match(stop_query, broader)
                                if closest:
                                    params['query'] = closest
                                    response = requests.get(url, params=params, timeout=10)
                                    if response.status_code == 200:
                                        matches = response.json().get('matches', [])
            
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
                
                route_was_specified = len(mentioned_routes) > 0
                
                if mentioned_routes and bus_stop_matches:
                    filtered = [m for m in bus_stop_matches
                                if any(r in [(l.get('id','').upper() if isinstance(l,dict) else str(l).upper()) for l in m.get('lines',[])]
                                       for r in mentioned_routes)]
                    if filtered:
                        matches = filtered
                    elif route_was_specified:
                        return {'error': 'route_not_served', 'query': stop_query,
                                'stop_name': bus_stop_matches[0].get('name', stop_query),
                                'requested_route': ', '.join(mentioned_routes),
                                'available_routes': [], 'available_routes_str': 'unknown'}
                
                # Simple disambiguation
                if len(bus_stop_matches) > 1 and not route_was_specified:
                    # Simple disambiguation: check if stops have different directions
                    towards_set = set((m.get('towards') or '').strip() for m in bus_stop_matches if (m.get('towards') or '').strip())
                    if len(towards_set) > 1:
                        labelled_stops = []
                        disambiguation_options = []
                        for match in bus_stop_matches:
                            stop_id = match.get('id', '')
                            name = match.get('name') or stop_query
                            towards = (match.get('towards') or '').strip()
                            platform = self._platform_from_stop_id(stop_id)
                            if platform and towards:
                                label = f"{name} – Stop {platform} (towards {towards})"
                            elif towards:
                                label = f"{name} (towards {towards})"
                            elif platform:
                                label = f"{name} – Stop {platform}"
                            else:
                                label = name
                            labelled_stops.append(label)
                            disambiguation_options.append({
                                'id': stop_id, 'name': name,
                                'towards': towards, 'platform': platform, 'label': label
                            })
                        return {
                            'error': 'disambiguation_needed',
                            'query': stop_query, 'mode': 'bus',
                            'stations': labelled_stops,
                            'count': len(labelled_stops),
                            'disambiguation_options': disambiguation_options
                        }
            
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

    # Map display names to TfL Line/Mode API ids
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
        
        # If no exact match, try the route name directly
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
                
                result = {
                    'route': line_name, 'status': status,
                    'delay_minutes': delay_minutes,
                    'timestamp': datetime.now().isoformat(), 'source': 'TFL'
                }
                if delay_minutes > 0:
                    result['congestion_level'] = 'light' if delay_minutes < 10 else 'moderate' if delay_minutes < 20 else 'heavy'
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
    
    
    