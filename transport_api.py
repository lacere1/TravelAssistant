"""
Transport Data Fetcher
Integrates with real-world transport APIs (TfL, Google Maps).
Timetables, line status, disruptions, route recommendations, stop info.
Returns None or empty list if no API data is available
"""
import requests
import os
import re
from typing import Dict, Optional, Any, List
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
