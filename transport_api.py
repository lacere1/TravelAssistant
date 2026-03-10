import os
import requests
import re
from datetime import datetime


class TransportDataFetcher:
    def __init__(self):
        self.tfl_app_id = os.environ.get('TFL_APP_ID', '')
        self.tfl_app_key = os.environ.get('TFL_APP_KEY', '')
        self.tfl_base_url = 'https://api.tfl.gov.uk'
        self.has_tfl = bool(self.tfl_app_id and self.tfl_app_key)

        if self.has_tfl:
            print("TFL API initialized")

    def _tfl_params(self):
        params = {}
        if self.tfl_app_id:
            params["app_id"] = self.tfl_app_id
        if self.tfl_app_key:
            params["app_key"] = self.tfl_app_key
        return params

    def _normalize_query(self, query):
        common_words = {
            'station', 'stop', 'the', 'tube', 'underground', 'london',
            'traffic', 'bus', 'train'
        }

        if not query or not query.strip():
            return '', [], None

        words = [w.lower().strip() for w in query.split() if w.strip()]
        meaningful = [w for w in words if w not in common_words]
        cleaned = ' '.join(meaningful)
        return cleaned, meaningful, meaningful[0] if meaningful else None

    def _normalize_tube_station_name(self, station_name):
        if not station_name or not station_name.strip():
            return station_name

        normalized = station_name.strip()
        normalized = re.sub(r'\bSt\.\b', 'St', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bst\b', 'St', normalized)
        normalized = re.sub(r'\s*&\s*', ' and ', normalized)
        normalized = re.sub(r'\s*-\s*', '-', normalized)
        return normalized.title()

    def _find_closest_location_match(self, query, candidates):
        from difflib import SequenceMatcher

        if not query or not candidates:
            return None

        best_match = None
        best_ratio = 0.0

        for candidate in candidates:
            ratio = SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate

        return best_match if best_ratio > 0.6 else None

    def _fetch_tfl_line_status(self, line_name):
        if not self.has_tfl or not line_name:
            return None

        try:
            url = f'{self.tfl_base_url}/line/{line_name}/status'
            response = requests.get(url, params=self._tfl_params(), timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    status = data[0].get('lineStatuses', [{}])[0].get('statusSeverityDescription', 'Unknown')
                    return {'status': status, 'line': line_name}
        except Exception as e:
            print(f"TFL API error: {e}")

        return None

    def get_route_traffic(self, route):
        if not route:
            return None

        traffic_data = self._fetch_tfl_line_status(route)
        return traffic_data

    _TRAIN_LINE_DISPLAY_TO_ID = {
        'central line': 'central',
        'northern line': 'northern',
        'bakerloo line': 'bakerloo',
        'circle line': 'circle',
        'district line': 'district',
    }

    def get_transit_disruption(self, line_name):
        if not self.has_tfl or not line_name:
            return None

        line_id = self._TRAIN_LINE_DISPLAY_TO_ID.get(line_name.lower(), line_name.lower())

        try:
            url = f'{self.tfl_base_url}/line/{line_id}/status'
            response = requests.get(url, params=self._tfl_params(), timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    statuses = data[0].get('lineStatuses', [])
                    disruptions = []
                    for status in statuses:
                        if status.get('statusSeverityDescription', '').lower() != 'good service':
                            disruptions.append({
                                'line': line_name,
                                'status': status.get('statusSeverityDescription', 'Unknown'),
                                'reason': status.get('reason', ''),
                            })
                    return disruptions if disruptions else None
        except Exception as e:
            print(f"TFL API error: {e}")

        return None

    def get_bus_disruption(self, route_name):
        if not self.has_tfl or not route_name:
            return None

        try:
            url = f'{self.tfl_base_url}/line/{route_name}/status'
            response = requests.get(url, params=self._tfl_params(), timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    statuses = data[0].get('lineStatuses', [])
                    disruptions = []
                    for status in statuses:
                        if status.get('statusSeverityDescription', '').lower() != 'good service':
                            disruptions.append({
                                'route': route_name,
                                'status': status.get('statusSeverityDescription', 'Unknown'),
                                'reason': status.get('reason', ''),
                            })
                    return disruptions if disruptions else None
        except Exception as e:
            print(f"TFL API error: {e}")

        return None

    def _build_timetable_from_arrivals(self, arrivals):
        timetable = []
        for arrival in arrivals[:5]:
            timetable.append({
                'line': arrival.get('lineId', 'Unknown'),
                'destination': arrival.get('destinationName', 'Unknown'),
                'time_to_station': arrival.get('timeToStation', 0),
                'expected_arrival': arrival.get('expectedArrival', ''),
            })
        return timetable

    def get_tfl_timetable(self, stop_name):
        if not self.has_tfl or not stop_name:
            return None

        try:
            stop_name_normalized = self._normalize_tube_station_name(stop_name)
            url = f'{self.tfl_base_url}/StopPoint/Search'
            params = self._tfl_params()
            params['query'] = stop_name_normalized
            params['modes'] = 'tube,bus'

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                if matches:
                    stop_id = matches[0].get('id')
                    arrival_url = f'{self.tfl_base_url}/StopPoint/{stop_id}/Arrivals'
                    arrival_response = requests.get(arrival_url, params=self._tfl_params(), timeout=5)
                    if arrival_response.status_code == 200:
                        arrivals = arrival_response.json()
                        timetable = self._build_timetable_from_arrivals(arrivals)
                        return timetable if timetable else None
        except Exception as e:
            print(f"TFL API error: {e}")

        return None
