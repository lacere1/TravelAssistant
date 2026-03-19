"""
TfL domain datasets for stop/line matching.

This module exists so that `ner_processor.py` can be used only as a fallback
when the LLM entity extractor is unavailable.

It provides:
  - bus stop names (from `bus_stops.csv`)
  - train station names (from `train_stops.csv`)
  - TfL bus route identifiers (from `tfl_bus_routes.txt`)
  - supported TfL line names (constant list)
"""

from __future__ import annotations

import csv
import os
import re
from typing import List, Set, Tuple


TRAIN_LINES: List[str] = [
    "Bakerloo",
    "Central",
    "Circle",
    "District",
    "Hammersmith & City",
    "Jubilee",
    "Metropolitan",
    "Northern",
    "Piccadilly",
    "Victoria",
    "Waterloo & City",
    "London Overground",
    "Windrush",
    "Lioness",
    "Mildmay",
    "Suffragette",
    "Weaver",
    "Liberty",
    "DLR",
    "Docklands Light Railway",
    "Elizabeth",
]


_CACHE: dict[str, object] = {}


def _base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_bus_stops(bus_path: str) -> List[str]:
    bus_names: List[str] = []
    if not os.path.exists(bus_path):
        return bus_names

    # Primary encoding
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
        return bus_names
    except UnicodeDecodeError:
        # Fallback encoding used in older exports
        with open(bus_path, newline="", encoding="cp1252") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = (row.get("CommonName") or "").strip()
                if raw_name:
                    name = re.sub(r"\s*\([^)]*\)\s*$", "", raw_name).strip()
                    if name:
                        bus_names.append(name)
        return bus_names


def _load_train_stations(train_path: str) -> List[str]:
    train_names: List[str] = []
    if not os.path.exists(train_path):
        return train_names

    def _get_station_name(row: dict) -> str:
        raw = (row.get("Station") or row.get("Stop") or row.get("Name") or "").strip()
        return re.sub(r"\s*\([^)]*\)\s*$", "", raw).strip() if raw else ""

    # Primary encoding
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

    # De-duplicate while preserving order
    return list(dict.fromkeys(train_names))


def _load_bus_routes(bus_routes_path: str) -> Set[str]:
    bus_route_ids: Set[str] = set()
    if not os.path.exists(bus_routes_path):
        return bus_route_ids

    with open(bus_routes_path, encoding="utf-8") as f:
        for line in f:
            rid = line.strip()
            if rid:
                bus_route_ids.add(rid)
                bus_route_ids.add(rid.upper())

    return bus_route_ids


def load_stop_datasets() -> Tuple[List[str], List[str], Set[str]]:
    """
    Load and cache bus stops, train stations, and TfL bus routes.

    Returns:
      (bus_stop_names, train_station_names, bus_route_ids)
    """
    cached = _CACHE.get("stop_datasets")
    if cached is not None:
        bus_stops, train_stations, bus_routes = cached  # type: ignore[misc]
        return bus_stops, train_stations, bus_routes

    base_dir = _base_dir()
    bus_path = os.path.join(base_dir, "bus_stops.csv")
    train_path = os.path.join(base_dir, "train_stops.csv")
    bus_routes_path = os.path.join(base_dir, "tfl_bus_routes.txt")

    bus_stops: List[str] = []
    train_stations: List[str] = []
    bus_routes: Set[str] = set()

    try:
        bus_stops = _load_bus_stops(bus_path)
    except Exception as e:
        print(f"[Datasets] Failed to load bus_stops.csv: {e}")

    try:
        train_stations = _load_train_stations(train_path)
    except Exception as e:
        print(f"[Datasets] Failed to load train_stops.csv: {e}")

    try:
        bus_routes = _load_bus_routes(bus_routes_path)
    except Exception as e:
        print(f"[Datasets] Failed to load tfl_bus_routes.txt: {e}")

    _CACHE["stop_datasets"] = (bus_stops, train_stations, bus_routes)
    print(
        f"[Datasets] Loaded {len(bus_stops)} bus stops, "
        f"{len(train_stations)} train stations, {len(bus_routes)} bus routes"
    )
    return bus_stops, train_stations, bus_routes


def load_bus_stops() -> List[str]:
    return load_stop_datasets()[0]


def load_train_stations() -> List[str]:
    return load_stop_datasets()[1]


def load_bus_routes() -> Set[str]:
    return load_stop_datasets()[2]

