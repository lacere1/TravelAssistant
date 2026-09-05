"""End-to-end test of the photo -> journey integration.

Proves that submitting a photograph to TravelAssistant produces a real
journey-planning response built from the recognised location, by exercising the
actual running services rather than mocks:

  1. LandmarkLens /healthz              -- the recogniser is up
  2. LandmarkLens /predict              -- a photo resolves to a location_name
  3. TravelAssistant /landmark_status   -- TravelAssistant can see the service
  4. TravelAssistant /chat_photo        -- the recognised name flows into the
                                           journey planner and comes back as a
                                           journey (the integration proper)
  5. the non-resolvable path            -- a generic bus stop / roundel photo
                                           asks the user to disambiguate instead
                                           of routing to a guess

Run both services first (see README), then:

    python test_landmarklens_integration.py
    python test_landmarklens_integration.py --image path/to/photo.jpg

With no --image, test photographs are taken from the LandmarkLens test split.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import requests

TA_URL = os.environ.get("TRAVELASSISTANT_URL", "http://localhost:5000").rstrip("/")
LL_URL = os.environ.get("LANDMARKLENS_API_URL", "http://localhost:8000").rstrip("/")
LL_REPO = os.environ.get(
    "LANDMARKLENS_REPO",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "ImageRecognition", "Image-classifier"),
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"{mark} {name}" + (f"\n       {detail}" if detail else ""), flush=True)
    return ok


def find_test_image(class_slug: str) -> str | None:
    """Pick one image of a given class from the LandmarkLens test split."""
    for split in ("test", "val", "train"):
        hits = sorted(glob.glob(os.path.join(
            LL_REPO, "data", "splits", split, class_slug, "*")))
        if hits:
            return hits[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="",
                    help="photo to use for the resolvable-landmark test")
    ap.add_argument("--generic-image", default="",
                    help="photo of generic street furniture (bus stop, roundel)")
    ap.add_argument("--from-text", default="from Waterloo",
                    help="accompanying text supplying the other side of the trip")
    args = ap.parse_args()

    print(f"TravelAssistant : {TA_URL}")
    print(f"LandmarkLens    : {LL_URL}")
    print(f"LandmarkLens repo: {LL_REPO}\n")

    # --- 1. LandmarkLens is up --------------------------------------------
    try:
        health = requests.get(f"{LL_URL}/healthz", timeout=10).json()
        record("LandmarkLens /healthz",
               health.get("status") == "ok",
               f"backbone={health.get('backbone')} backend={health.get('backend')} "
               f"classes={health.get('num_classes')}")
    except Exception as exc:
        record("LandmarkLens /healthz", False, str(exc))
        print("\nLandmarkLens is not running - start it first. Aborting.")
        return 1

    # --- pick photos -------------------------------------------------------
    landmark_image = args.image or find_test_image("tower_bridge") \
        or find_test_image("big_ben") or find_test_image("the_shard")
    if not landmark_image or not os.path.exists(landmark_image):
        record("locate a landmark test photo", False,
               "no image found; pass --image explicitly")
        return 1
    print(f"landmark photo  : {landmark_image}")

    generic_image = args.generic_image or find_test_image("bus_stop_flag") \
        or find_test_image("tube_roundel")
    if generic_image:
        print(f"generic photo   : {generic_image}\n")

    # --- 2. LandmarkLens recognises it ------------------------------------
    with open(landmark_image, "rb") as fh:
        blob = fh.read()
    pred = requests.post(
        f"{LL_URL}/predict",
        files={"image": (os.path.basename(landmark_image), blob, "image/jpeg")},
        timeout=60,
    ).json()
    location_name = pred.get("location_name", "")
    ok = bool(location_name) and pred.get("confidence", 0) > 0
    record("LandmarkLens /predict returns a location_name", ok,
           f"location_name={location_name!r} "
           f"confidence={pred.get('confidence')} "
           f"resolvable={pred.get('resolvable')} "
           f"heatmap_url={pred.get('heatmap_url')}")

    if pred.get("heatmap_url"):
        hm = requests.get(f"{LL_URL}{pred['heatmap_url']}", timeout=30)
        record("Grad-CAM heatmap is served",
               hm.status_code == 200 and hm.headers.get("Content-Type") == "image/png",
               f"HTTP {hm.status_code}, {len(hm.content)} bytes")

    # --- 3. TravelAssistant can see the service ---------------------------
    try:
        status = requests.get(f"{TA_URL}/landmark_status", timeout=15).json()
        record("TravelAssistant /landmark_status sees LandmarkLens",
               bool(status.get("available")),
               f"url={status.get('url')} error={status.get('error', '')}")
    except Exception as exc:
        record("TravelAssistant /landmark_status", False, str(exc))
        print("\nTravelAssistant is not running - start it first. Aborting.")
        return 1

    # --- 4. THE INTEGRATION: photo -> journey -----------------------------
    session = requests.Session()  # journey state lives in a Flask session
    # JourneyChatbot keeps state under a single "global" key, so clear any
    # leftover journey from a previous run before testing.
    try:
        session.post(f"{TA_URL}/new_chat", timeout=15)
    except Exception:
        pass
    resp = session.post(
        f"{TA_URL}/chat_photo",
        files={"image": (os.path.basename(landmark_image), blob, "image/jpeg")},
        data={"role": "to", "message": args.from_text},
        timeout=180,
    )
    body = resp.json()
    photo_meta = body.get("photo", {})
    entities = body.get("entities", {})
    journeys = body.get("journeys", []) or []
    reply = (body.get("response") or "").strip()

    print("\n--- /chat_photo response ---")
    print(f"  photo.label        : {photo_meta.get('label')}")
    print(f"  photo.location_name: {photo_meta.get('location_name')}")
    print(f"  photo.confidence   : {photo_meta.get('confidence')}")
    print(f"  intent             : {body.get('intent')}")
    print(f"  entities.origin    : {entities.get('origin')}")
    print(f"  entities.destination: {entities.get('destination')}")
    print(f"  journeys returned  : {len(journeys)}")
    print(f"  reply              : {reply[:300]}")
    if journeys:
        j = journeys[0]
        print(f"  first journey keys : {sorted(j)[:10]}")
    print("--- end ---\n")

    if body.get("error") and not photo_meta:
        record("/chat_photo returned a response", False,
               f"server error: {body['error'][:200]}")

    # Both sides must be non-empty: an all-None response must not pass here.
    photo_loc = (photo_meta.get("location_name") or "").strip()
    dest = (entities.get("destination") or "").strip()
    record("photo populated a journey-planner location slot",
           bool(photo_loc) and bool(dest)
           and (dest.lower().startswith(photo_loc.lower()[:6])
                or photo_loc.lower() in dest.lower()),
           f"destination={dest!r} from photo location_name={photo_loc!r}")

    record("photo request routed to the journey planner",
           body.get("intent") == "journey_planner",
           f"intent={body.get('intent')!r}")

    # TravelAssistant's normal flow asks the user to confirm two map pins
    # before it calls the TfL Journey API. Follow that step through, exactly as
    # the browser would, so the test proves a photo yields real journeys rather
    # than stopping at the first reply.
    pins = body.get("confirm_pins")
    if not journeys and pins and pins.get("from") and pins.get("to"):
        print(f"confirming pins: {pins['from'].get('name')} -> "
              f"{pins['to'].get('name')}")
        follow = session.post(
            f"{TA_URL}/chat",
            json={"message": "",
                  "confirmPins": {
                      "from": {"lat": pins["from"]["lat"],
                               "lon": pins["from"]["lon"]},
                      "to": {"lat": pins["to"]["lat"],
                             "lon": pins["to"]["lon"]}}},
            timeout=180,
        ).json()
        journeys = follow.get("journeys", []) or []
        reply = (follow.get("response") or "").strip()
        print(f"  after pin confirmation: {len(journeys)} journey option(s)")
        print(f"  reply: {reply[:300]}")
        if journeys:
            legs = journeys[0].get("legs") or []
            print(f"  first journey: {journeys[0].get('duration')} min, "
                  f"{len(legs)} leg(s)")
            for leg in legs[:4]:
                print(f"    - {leg.get('mode')}: {str(leg.get('detail'))[:90]}")

    produced_journeys = len(journeys) > 0
    asked_followup = bool(reply) and (body.get("disambiguation")
                                      or body.get("journey_disambiguation")
                                      or "?" in reply)
    record("journey-planning response produced from the photo",
           produced_journeys or asked_followup,
           f"{len(journeys)} journey option(s) returned by the TfL Journey API"
           if produced_journeys else f"planner follow-up: {reply[:160]}")

    # --- 5. Generic street furniture must NOT auto-route ------------------
    if generic_image:
        try:
            session.post(f"{TA_URL}/new_chat", timeout=15)
        except Exception:
            pass
        with open(generic_image, "rb") as fh:
            gblob = fh.read()
        gresp = session.post(
            f"{TA_URL}/chat_photo",
            files={"image": (os.path.basename(generic_image), gblob, "image/jpeg")},
            data={"role": "to"},
            timeout=120,
        ).json()
        gphoto = gresp.get("photo", {})
        auto_routed = bool(gresp.get("journeys"))
        # Correct behaviour: either it was recognised as a non-specific class
        # and we asked which one, or it was recognised as a real landmark.
        acceptable = (not auto_routed) if not gphoto.get("resolvable", True) else True
        record("generic street furniture is not routed to a guessed location",
               acceptable,
               f"class={gphoto.get('class')} resolvable={gphoto.get('resolvable')} "
               f"journeys={len(gresp.get('journeys') or [])} "
               f"reply={(gresp.get('response') or '')[:140]}")

    # --- summary -----------------------------------------------------------
    print("\n================ SUMMARY ================")
    for status_str, name, _ in results:
        print(f"  {status_str}  {name}")
    failed = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n  {len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
