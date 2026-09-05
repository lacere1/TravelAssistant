"""Client for the LandmarkLens image-recognition service.

TravelAssistant resolves a location from a photo by POSTing the image to a
LandmarkLens instance over HTTP. The two projects stay decoupled: nothing here
imports LandmarkLens internals, and if the service is down or slow the caller
gets a clean failure rather than a stack trace.

Configuration (environment / .env):
    LANDMARKLENS_API_URL      base URL, default http://localhost:8000
    LANDMARKLENS_TIMEOUT      per-request timeout in seconds, default 20
    LANDMARKLENS_MIN_CONF     confidence below which a prediction is treated
                              as too weak to act on, default 0.45
"""
from __future__ import annotations

import mimetypes
import os
import uuid
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

import requests


def api_url() -> str:
    return os.environ.get("LANDMARKLENS_API_URL", "http://localhost:8000").rstrip("/")


def _timeout() -> float:
    try:
        return float(os.environ.get("LANDMARKLENS_TIMEOUT", "20"))
    except ValueError:
        return 20.0


def _min_confidence() -> float:
    try:
        return float(os.environ.get("LANDMARKLENS_MIN_CONF", "0.45"))
    except ValueError:
        return 0.45


class LandmarkLensError(RuntimeError):
    """Raised when the recognition service cannot be reached or fails."""


@dataclass
class LandmarkResult:
    """One recognition outcome, in TravelAssistant's own terms."""

    location_name: str
    confidence: float
    label: str
    class_slug: str
    # True when `location_name` names a specific, geocodable place. False for
    # generic street furniture (a bus stop, a roundel), where the photo tells
    # us the *kind* of place but not which one -- the user must still say.
    resolvable: bool
    low_confidence: bool
    heatmap_url: Optional[str] = None
    raw: Optional[dict] = None

    @property
    def usable(self) -> bool:
        """Safe to feed straight into the journey pipeline as a location."""
        return self.resolvable and not self.low_confidence

    def describe(self) -> str:
        pct = f"{self.confidence * 100:.0f}%"
        return f"{self.label} ({pct} confidence)"


def recognise(image_bytes: bytes, filename: str = "upload.jpg",
              want_heatmap: bool = True) -> LandmarkResult:
    """Send an image to LandmarkLens and return the recognised location.

    Raises LandmarkLensError on transport failure or a non-2xx response.
    """
    if not image_bytes:
        raise LandmarkLensError("no image data supplied")

    base = api_url()
    url = urljoin(base + "/", "predict")
    if not want_heatmap:
        url += "?heatmap=false"

    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    try:
        resp = requests.post(
            url,
            files={"image": (filename or "upload.jpg", image_bytes, mime)},
            timeout=_timeout(),
        )
    except requests.exceptions.ConnectionError as exc:
        raise LandmarkLensError(
            f"could not reach the image recognition service at {base} "
            f"(is LandmarkLens running?)"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LandmarkLensError(
            f"image recognition timed out after {_timeout():.0f}s") from exc
    except requests.exceptions.RequestException as exc:
        raise LandmarkLensError(f"image recognition request failed: {exc}") from exc

    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("error", "")
        except Exception:
            detail = resp.text[:200]
        raise LandmarkLensError(
            f"image recognition failed (HTTP {resp.status_code}): {detail}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise LandmarkLensError("image recognition returned a non-JSON response") from exc

    location_name = (data.get("location_name") or "").strip()
    if not location_name:
        raise LandmarkLensError("image recognition returned no location_name")

    confidence = float(data.get("confidence") or 0.0)
    # `low_confidence` is computed here as well as by the service so the
    # threshold can be tuned from TravelAssistant's own config.
    low = bool(data.get("low_confidence")) or confidence < _min_confidence()

    heatmap = data.get("heatmap_url")
    if heatmap and heatmap.startswith("/"):
        heatmap = base + heatmap  # make the relative URL usable by the browser

    return LandmarkResult(
        location_name=location_name,
        confidence=confidence,
        label=data.get("label") or location_name,
        class_slug=data.get("class") or "",
        resolvable=bool(data.get("resolvable", True)),
        low_confidence=low,
        heatmap_url=heatmap,
        raw=data,
    )


def health() -> dict:
    """Ping the service. Returns its /healthz payload, or raises."""
    try:
        resp = requests.get(urljoin(api_url() + "/", "healthz"), timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise LandmarkLensError(f"LandmarkLens not reachable: {exc}") from exc


def is_available() -> bool:
    try:
        health()
        return True
    except LandmarkLensError:
        return False
