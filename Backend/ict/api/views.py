from datetime import datetime, date, timedelta, timezone
from collections import Counter
import random
import json
import logging
import threading
import re
import pprint
import copy
import time
import base64
import io
import os

from django.core.exceptions import ObjectDoesNotExist
from django.core.mail import send_mail
from django.db import IntegrityError, transaction as db_transaction
from django.db.models import F, Value, CharField
from django.db.models.functions import Concat
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.conf import settings
from django.utils.timezone import now

from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.views import ObtainAuthToken

from .models import *
from .serializers import *
from .utils import *

import requests
from django.utils.html import escape

import numpy as np
from PIL import Image

import face_recognition
from ultralytics import YOLO

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple
from .hand_recognition import HandLandmarkService
hand_service = HandLandmarkService()  # singleton-ish

_DETECTOR = None
# ------------------------------------------------------------------------------
# YOLO (object detection) - keep your existing singleton if you want
# ------------------------------------------------------------------------------
YOLO_MODEL = YOLO("yolov8n.pt")  # process-level singleton

# ------------------------------------------------------------------------------
# MediaPipe Models (Object + Gesture)
# ------------------------------------------------------------------------------
MP_MODEL_DIR = Path(settings.BASE_DIR) / "model"

# EfficientDet object detector bundle
MP_OBJECT_MODEL_PATH = str(MP_MODEL_DIR / "efficientdet_lite2.tflite")

# Gesture recognizer task bundle
MP_GESTURE_MODEL_PATH = str(MP_MODEL_DIR / "gesture_recognizer.task")

# Monotonic baseline for VIDEO timestamps (object detector)
START_TIME = time.monotonic()

# ---- Object detector: create ONE instance (VIDEO mode) ----
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

_DETECTOR_LOCK = threading.Lock()

_DETECTOR_OPTIONS = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=MP_OBJECT_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    max_results=5,
    score_threshold=0.30,
)

DETECTOR_INSTANCE = ObjectDetector.create_from_options(_DETECTOR_OPTIONS)

# ---- Gesture recognizer: lazy singleton (IMAGE mode) ----
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRunningMode = mp.tasks.vision.RunningMode

_GESTURE_LOCK = threading.Lock()
_GESTURE = None


def get_gesture_recognizer():
    """Return a cached MediaPipe GestureRecognizer (IMAGE mode)."""
#    global _GESTURE
#    if _GESTURE is not None:
#        return _GESTURE
#
#    with _GESTURE_LOCK:
#        if _GESTURE is not None:
#            return _GESTURE
#
#        if not os.path.isfile(MP_GESTURE_MODEL_PATH):
#            raise FileNotFoundError(f"Missing gesture model: {MP_GESTURE_MODEL_PATH}")
#
#        options = GestureRecognizerOptions(
#            base_options=BaseOptions(model_asset_path=MP_GESTURE_MODEL_PATH),
#            running_mode=GestureRunningMode.IMAGE,
#        )
#
#        _GESTURE = GestureRecognizer.create_from_options(options)
#        return _GESTURE
    global _GESTURE
    if _GESTURE is not None:
        return _GESTURE

    with _GESTURE_LOCK:
        if _GESTURE is not None:
            return _GESTURE

        if not os.path.isfile(MP_GESTURE_MODEL_PATH):
            raise FileNotFoundError(f"Missing gesture model: {MP_GESTURE_MODEL_PATH}")

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=MP_GESTURE_MODEL_PATH),
            running_mode=GestureRunningMode.IMAGE,
        )

        _GESTURE = GestureRecognizer.create_from_options(options)
        return _GESTURE


# ------------------------------------------------------------------------------
# YOLO helper (fixed: no missing globals)
# ------------------------------------------------------------------------------
_YOLO_X_MODEL = None
_YOLO_X_LOCK = threading.Lock()

def get_model():
    """
    Cached YOLO model loader (yolov8x.pt).
    """
    global _YOLO_X_MODEL
    if _YOLO_X_MODEL is None:
        with _YOLO_X_LOCK:
            if _YOLO_X_MODEL is None:
                _YOLO_X_MODEL = YOLO("yolov8x.pt")
    return _YOLO_X_MODEL


def _clamp(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def _normalize_label(s: str) -> str:
    return (s or "").strip().lower()


def _xyxy_to_norm_xyxy(xyxy, w, h):
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = (area_a + area_b - inter)
    return float(inter / union) if union > 0 else 0.0


def _nms(dets, iou_thresh=0.50):
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        if any(_iou(d["box"], k["box"]) >= iou_thresh for k in keep):
            continue
        keep.append(d)
    return keep


@csrf_exempt
def hand_recognition(request):
    """
    Hand gesture recognition endpoint.

    Supports BOTH payload styles:

    1) multipart/form-data (legacy):
       - form field: image=<file>

    2) application/json (your browser client):
       {
         "session_id": "...",
         "frame_id": 1,
         "ts_client_ms": 123,
         "image": { "format": "jpeg", "data_b64": "<BASE64_NO_PREFIX>" },
         "telemetry": { ... }
       }

    Returns top gesture + handedness.
    """

    if request.method == "OPTIONS":
        return HttpResponse(status=204)

    if request.method != "POST":
        return JsonResponse({
            "ok": False,
            "error": {"code": "METHOD_NOT_ALLOWED", "message": "POST only"}
        }, status=405)

    t0 = time.perf_counter()

    # -----------------------------
    # 1) Try multipart/form-data first
    # -----------------------------
    uploaded = request.FILES.get("image")
    if uploaded:
        try:
            uploaded.seek(0)
        except Exception:
            pass

        data = uploaded.read()
        if not data:
            return JsonResponse({
                "ok": False,
                "error": {"code": "IMAGE_EMPTY", "message": "Uploaded image is empty."}
            }, status=400)

        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return JsonResponse({
                "ok": False,
                "error": {"code": "IMAGE_DECODE_FAILED", "message": "cv2.imdecode returned None"}
            }, status=400)

        # no session metadata for multipart requests
        session_id = None
        frame_id = None
        ts_client_ms = None
        telemetry = {}

    else:
        # -----------------------------
        # 2) JSON body (base64 JPEG)
        # -----------------------------
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except Exception as e:
            return JsonResponse({
                "ok": False,
                "error": {"code": "BAD_JSON", "message": f"Invalid JSON body: {str(e)}"}
            }, status=400)

        session_id = payload.get("session_id")
        frame_id = payload.get("frame_id")
        ts_client_ms = payload.get("ts_client_ms")
        telemetry = payload.get("telemetry") or {}

        image_obj = payload.get("image") or {}
        data_b64 = image_obj.get("data_b64")

        if not data_b64:
            return JsonResponse({
                "ok": False,
                "error": {"code": "IMAGE_MISSING", "message": "Missing JSON field image.data_b64 (or form-data field 'image')."}
            }, status=400)

        try:
            # Accept both bare base64 and data URLs
            if "base64," in data_b64:
                data_b64 = data_b64.split("base64,", 1)[1]
            raw = base64.b64decode(data_b64, validate=False)
            arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            return JsonResponse({
                "ok": False,
                "error": {"code": "IMAGE_DECODE_FAILED", "message": f"Base64 decode failed: {str(e)}"}
            }, status=400)

    # -----------------------------
    # Run MediaPipe gesture recognizer
    # -----------------------------
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        recognizer = get_gesture_recognizer()
        # recognizer is not documented as thread-safe; lock during inference
        with _GESTURE_LOCK:
            # If the caller provided a timestamp, prefer VIDEO inference.
            # Otherwise fall back to IMAGE inference.
            if ts_client_ms is not None:
                try:
                    ts_ms = int(ts_client_ms)
                except Exception:
                    ts_ms = int(time.time() * 1000)
                try:
                    result = recognizer.recognize_for_video(mp_image, ts_ms)
                except Exception:
                    # if recognizer was created in IMAGE mode, this will fail
                    result = recognizer.recognize(mp_image)
            else:
                result = recognizer.recognize(mp_image)

    except FileNotFoundError as e:
        return JsonResponse({
            "ok": False,
            "error": {"code": "NOT_CONFIGURED", "message": str(e)}
        }, status=500)

    except Exception as e:
        return JsonResponse({
            "ok": False,
            "error": {"code": "GESTURE_INFERENCE_FAILED", "message": str(e)}
        }, status=500)

    gesture_name = None
    gesture_score = 0.0
    if getattr(result, "gestures", None) and result.gestures and len(result.gestures[0]) > 0:
        top = result.gestures[0][0]
        gesture_name = top.category_name
        gesture_score = float(top.score or 0.0)

    hand_name = None
    hand_score = 0.0
    if getattr(result, "handedness", None) and result.handedness and len(result.handedness[0]) > 0:
        h0 = result.handedness[0][0]
        hand_name = h0.category_name
        hand_score = float(h0.score or 0.0)

        # If your UI is mirrored (front camera), swap labels to match what the user sees.
        # NOTE: Your browser client already mirrors capture when mirror=true.
        # If you want server-side mirroring, do it only when telemetry.mirror is true.
        if telemetry.get("mirror") is True:
            if hand_name == "Left":
                hand_name = "Right"
            elif hand_name == "Right":
                hand_name = "Left"

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return JsonResponse({
        "ok": True,
        "session_id": session_id,
        "frame_id": frame_id,
        "ts_server_ms": int(time.time() * 1000),
        "latency_ms": latency_ms,
        "gesture": {"name": gesture_name, "score": gesture_score},
        "handedness": {"name": hand_name, "score": hand_score},
        "telemetry": telemetry,
    })

# ------------------------------------------------------------------------------
# Face recognition tool (fixed: single decode method)
# ------------------------------------------------------------------------------
class FacialRecognitionTool:
    def __init__(self):
        self.KNOWN_FACES_DIR = os.path.join(settings.BASE_DIR, "known_faces")
        self.TOLERANCE = 0.50
        self.FRAME_SCALE = 0.50

    def ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _decode_uploaded_image(self, uploaded_file):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        data = uploaded_file.read()
        if not data:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def load_facial_database(self):
        known_encodings, known_names = [], []

        if not os.path.isdir(self.KNOWN_FACES_DIR):
            self.ensure_dir(self.KNOWN_FACES_DIR)
            return known_encodings, known_names

        try:
            for name in sorted(os.listdir(self.KNOWN_FACES_DIR)):
                person_dir = os.path.join(self.KNOWN_FACES_DIR, name)
                if not os.path.isdir(person_dir):
                    continue
                enc_path = os.path.join(person_dir, "encodings.npy")
                if not os.path.exists(enc_path):
                    continue
                encs = np.load(enc_path)
                for e in encs:
                    known_encodings.append(e)
                    known_names.append(name)
        except Exception:
            return [], []

        return known_encodings, known_names

    def save_encoding(self, name: str, encoding: np.ndarray):
        person_dir = os.path.join(self.KNOWN_FACES_DIR, name)
        self.ensure_dir(person_dir)
        enc_path = os.path.join(person_dir, "encodings.npy")

        if os.path.exists(enc_path):
            encs = np.load(enc_path)
            encs = np.vstack([encs, encoding])
        else:
            encs = np.array([encoding])

        np.save(enc_path, encs)
        return int(encs.shape[0])

    def best_match(self, known_encodings, known_names, face_encoding):
        if not known_encodings:
            return "Unknown", 999.0
        dists = face_recognition.face_distance(known_encodings, face_encoding)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        return (known_names[idx], dist) if dist <= self.TOLERANCE else ("Unknown", dist)

    def recognize_from_upload(self, uploaded_file):
        frame = self._decode_uploaded_image(uploaded_file)
        if frame is None:
            return {"recognized": False, "name": "Unknown", "distance": 999.0, "error": "Invalid image"}

        small = cv2.resize(frame, (0, 0), fx=self.FRAME_SCALE, fy=self.FRAME_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model="hog")
        if not locations:
            return {"recognized": False, "name": "Unknown", "distance": 999.0}

        encodings = face_recognition.face_encodings(rgb_small, locations)
        if not encodings:
            return {"recognized": False, "name": "Unknown", "distance": 999.0}

        areas = [(b - t) * (r - l) for (t, r, b, l) in locations]
        i = int(np.argmax(areas))
        chosen_enc = encodings[i]

        known_encodings, known_names = self.load_facial_database()
        name, dist = self.best_match(known_encodings, known_names, chosen_enc)

        return {"recognized": name != "Unknown", "name": name, "distance": dist, "_encoding": chosen_enc}

    def enroll_from_upload(self, uploaded_file, name: str):
        result = self.recognize_from_upload(uploaded_file)
        enc = result.get("_encoding")
        if enc is None:
            return {"ok": False, "error": "No face detected"}
        samples_total = self.save_encoding(name, enc)
        return {"ok": True, "enrolled": True, "name": name, "samples_total": samples_total}


tool = FacialRecognitionTool()


@csrf_exempt
def recognize(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=405)

    uploaded = request.FILES.get("image")
    if not uploaded:
        return JsonResponse({"ok": False, "error": "Missing form-data field: image"}, status=400)

    result = tool.recognize_from_upload(uploaded)
    if "error" in result:
        return JsonResponse({"ok": False, "error": result["error"]}, status=400)

    return JsonResponse({
        "ok": True,
        "recognized": bool(result["recognized"]),
        "name": result["name"],
        "distance": float(result["distance"]),
        "timestamp": now().isoformat(),
    })


@csrf_exempt
def enroll(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=405)

    uploaded = request.FILES.get("image")
    name = (request.POST.get("name") or "").strip()

    if not uploaded:
        return JsonResponse({"ok": False, "error": "Missing form-data field: image"}, status=400)
    if not name:
        return JsonResponse({"ok": False, "error": "Missing form-data field: name"}, status=400)

    safe = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe:
        return JsonResponse({"ok": False, "error": "Invalid name"}, status=400)

    out = tool.enroll_from_upload(uploaded, safe)
    if not out.get("ok"):
        return JsonResponse({"ok": False, "error": out.get("error", "Enroll failed")}, status=400)

    return JsonResponse({
        "ok": True,
        "enrolled": True,
        "name": out["name"],
        "samples_total": out["samples_total"],
        "timestamp": now().isoformat(),
    })


def decode_b64_jpeg_to_bgr(data_b64: str) -> np.ndarray:
    """
    Supports either pure base64 or data URL format.
    Returns OpenCV BGR image.
    """
    if not data_b64:
        raise ValueError("Missing base64 data")

    if "base64," in data_b64:
        data_b64 = data_b64.split("base64,", 1)[1]

    raw = base64.b64decode(data_b64, validate=False)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid JPEG (cv2.imdecode returned None)")
    return bgr


def to_contract_objects(detection_result, w: int, h: int):
    """
    Convert MediaPipe detections into your frontend contract:
    [{ name, score, box:{x1,y1,x2,y2} }, ...] with normalized coords.
    """
    objects = []
    if not detection_result or not getattr(detection_result, "detections", None):
        return objects

    for det in detection_result.detections:
        if not det.categories:
            continue

        cat = det.categories[0]
        name = cat.category_name or "object"
        score = float(cat.score or 0.0)

        bbox = det.bounding_box
        x1 = float(bbox.origin_x)
        y1 = float(bbox.origin_y)
        x2 = float(bbox.origin_x + bbox.width)
        y2 = float(bbox.origin_y + bbox.height)

        # clamp
        x1 = max(0.0, min(x1, w))
        x2 = max(0.0, min(x2, w))
        y1 = max(0.0, min(y1, h))
        y2 = max(0.0, min(y2, h))

        objects.append({
            "name": name,
            "score": round(score, 4),
            "box": {
                "x1": round(x1 / w, 6),
                "y1": round(y1 / h, 6),
                "x2": round(x2 / w, 6),
                "y2": round(y2 / h, 6),
            }
        })

    objects.sort(key=lambda o: o["score"], reverse=True)
    return objects


# ------------------------------------------------------------------------------
# Existing endpoints (kept as-is unless obviously broken)
# ------------------------------------------------------------------------------
@api_view(['GET', 'POST'])
def test(request):
    Candle.objects.all().delete()
    return Response("successful")


@api_view(['GET'])
def get_data(request):
    instrument = request.GET.get('instrument', False)
    instrument_id = Instrument.objects.get(name=instrument)
    candles = Candle.objects.filter(instrument=instrument_id)
    response = CandleSerializers(candles, many=True).data
    return Response(response)


@api_view(['GET'])
def get_candles(request):
    instrument = request.GET.get('instrument')
    timeframe = request.GET.get('timeframe')
    amount_of_candles = request.GET.get('amount')
    instrument_obj = Instrument.objects.get(name=instrument)

    candles = Candle.objects.filter(
        instrument=instrument_obj,
        timeframe=timeframe
    )

    data = CandleSerializers(candles, many=True).data

    candle_set = []
    temp_array = []
    for i in data:
        if len(temp_array) == int(amount_of_candles):
            candle_set.append(copy.deepcopy(temp_array))
            temp_array.clear()
        else:
            temp_array.append(i)

    if not candle_set:
        return Response({"error": "Not enough candles"}, status=400)

    random_index = random.randint(0, len(candle_set) - 1)
    selected = candle_set[random_index]
    return Response(selected)


@api_view(['POST'])
def check_fvg(request):
    candles = request.data.get("candles", request.data)

    if not isinstance(candles, list):
        return Response({"detail": "Expected a list of 3 candles..."}, 400)

    if len(candles) != 3:
        return Response({"detail": "FVG check requires exactly 3 candles."}, 400)

    try:
        result = detect_fvg_from_array(candles)
    except Exception as e:
        return Response({"detail": f"Error: {str(e)}"}, 400)

    if result is None:
        return Response({
            "is_fvg": False,
            "type": None,
            "gap_high": None,
            "gap_low": None,
        })

    return Response({
        "is_fvg": True,
        "type": result["type"],
        "gap_high": result["gap_high"],
        "gap_low": result["gap_low"],
    })


@api_view(["POST"])
def check_order_block(request):
    direction = request.data.get("direction")
    candles = request.data.get("candles", [])
    require_displacement = request.data.get("require_displacement", True)
    min_body_ratio = request.data.get("min_body_ratio", 0.45)

    if not isinstance(candles, list) or len(candles) < 2:
        return Response({"ok": False, "matches": [], "reason": "need at least 2 candles"})

    if direction not in ("bullish", "bearish"):
        return Response({"ok": False, "matches": [], "reason": "invalid direction"})

    matches = find_order_blocks(
        candles,
        direction=direction,
        require_displacement=require_displacement,
        min_body_ratio=min_body_ratio,
        mss_lookback=3,
        ob_lookback=4,
        min_ob_body=0.25,
    )

    if not matches:
        return Response({"ok": False, "matches": [], "reason": "no order block found"})

    return Response({"ok": True, "matches": matches})


@api_view(["POST"])
def check_manual_ob(request):
    candles = request.data.get("candles", [])

    if not isinstance(candles, list) or len(candles) < 5:
        return Response({
            "ok": False,
            "is_ob": False,
            "direction": None,
            "prev_direction": "none",
            "reason": "need at least 5 candles (expected ~7)",
        })

    result = confirm_manual_ob(candles)
    return Response(result)


@api_view(["POST"])
def check_breaker_block(request):
    direction = request.data.get("direction")
    candles = request.data.get("candles", [])

    if not isinstance(candles, list) or len(candles) < 5:
        return Response({"ok": False, "reason": "need at least 5 candles"}, status=400)

    candles = candles[-12:]

    if direction == "bullish":
        ok, info = is_bullish_breaker(candles)
    elif direction == "bearish":
        ok, info = is_bearish_breaker(candles)
    else:
        return Response({"ok": False, "reason": "invalid direction"}, status=400)

    return Response({"ok": ok, "info": info})


@csrf_exempt
@api_view(['POST'])
def login_verification(request):
    try:
        serializer = ObtainAuthToken.serializer_class(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']

        profile_access_roles = user.profile_access.all().values_list('name', flat=True)

        token, created = Token.objects.get_or_create(user=user)

        return Response({
            'token': token.key,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'username': user.username,
            'email': user.email,
            'active': user.is_active,
            'profile_access': list(profile_access_roles)
        })

    except User.DoesNotExist:
        return Response({"error": "User does not exist"}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        print(e)
        return Response({"errors": "An unexpected error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET", "POST"])
def trade_journal_view(request):
    if request.method == "POST":
        serializer = TradeJournalSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    trade_id = request.query_params.get("trade_id")
    external_trade_id = request.query_params.get("external_trade_id")

    if trade_id is not None:
        trade = get_object_or_404(TradeJournal, pk=trade_id)
        serializer = TradeJournalSerializer(trade)
        return Response(serializer.data, status=status.HTTP_200_OK)

    if external_trade_id is not None:
        trade = get_object_or_404(TradeJournal, external_trade_id=external_trade_id)
        serializer = TradeJournalSerializer(trade)
        return Response(serializer.data, status=status.HTTP_200_OK)

    user = request.GET.get("username")
    if not user:
        return Response({"detail": "Username is required."}, status=400)

    try:
        userInstance = CustomUser.objects.get(username=user)
    except CustomUser.DoesNotExist:
        return Response({"detail": "User not found."}, status=404)

    trades = TradeJournal.objects.filter(username=userInstance).order_by("-date", "-time")
    serializer = TradeJournalSerializer(trades, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(["GET", "POST"])
def flashcard(request):
    if request.method == "GET":
        serializer = FlashCardSerializers(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_flashcard(request):
    course = request.GET.get('course', False)
    cards = Flashcard.objects.filter(course=course)
    response = FlashCardSerializers(cards, many=True).data
    return Response(response)

@csrf_exempt  # simplest for local dev; see CSRF notes below for production
def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    if "image" not in request.FILES:
        return JsonResponse({"error": "Missing file field 'image'."}, status=400)

    # Confidence threshold
    try:
        conf = float(request.POST.get("conf", "0.25"))
    except ValueError:
        conf = 0.15

    # Read uploaded image into numpy array (RGB)
    uploaded = request.FILES["image"].read()
    img = Image.open(io.BytesIO(uploaded)).convert("RGB")
    frame = np.array(img)  # shape: (H, W, 3) RGB
    model = get_model()

    # Run inference (no saving)
    results = model.predict(
        frame,
        conf=conf,
        iou=0.5,
        max_det=300,
        imgsz=960,
        verbose=False
    )
    r = results[0]
    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_id": cls_id,
                "class_name": model.names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf[0]),
                "bbox_xyxy": [float(x) for x in box.xyxy[0].tolist()],  # [x1,y1,x2,y2]
            })

    return JsonResponse({"detections": detections})


import base64
import binascii
import cv2
import numpy as np


def _decode_b64_jpeg(b64: str) -> np.ndarray:
    """
    Decode a base64-encoded JPEG into an OpenCV BGR image.

    Accepts:
      - raw base64 (no prefix)
      - data URLs like: data:image/jpeg;base64,...

    Returns:
      - np.ndarray (BGR image)

    Raises:
      - ValueError with a clear message if decoding fails
    """
    if not b64 or not isinstance(b64, str):
        raise ValueError("image.data_b64 missing or invalid")

    # Strip data URL prefix if present
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("Invalid base64 image data") from e

    np_buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode JPEG (cv2.imdecode returned None)")

    return img
    
# -----------------------------
def decode_image(uploaded_file):
    data = uploaded_file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img

def _err(status: int, code: str, message: str, extra: Optional[Dict[str, Any]] = None):
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if extra:
        payload.update(extra)
    return JsonResponse(payload, status=status)


def _get_detector():
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    with _DETECTOR_LOCK:
        if _DETECTOR is not None:
            return _DETECTOR

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=MP_OBJECT_MODEL_PATH),
            running_mode=RunningMode.IMAGE,   # IMPORTANT: server requests are independent frames
            max_results=8,
            score_threshold=0.30,
        )
        _DETECTOR = ObjectDetector.create_from_options(options)
        return _DETECTOR


def _json_error(status: int, code: str, message: str, extra=None):
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if extra:
        payload["error"]["extra"] = extra
    return JsonResponse(payload, status=status)


@csrf_exempt
def object_gesture(request):
    if request.method == "OPTIONS":
        return HttpResponse(status=204)

    if request.method != "POST":
        return _json_error(405, "POST_REQUIRED", "POST required")

    t0 = time.time()

    try:
        payload = json.loads(request.body.decode("utf-8"))
        session_id = payload["session_id"]
        frame_id = payload["frame_id"]
        ts_client_ms = payload["ts_client_ms"]
        focus_label = payload.get("focus_label")
        telemetry = payload.get("telemetry", {})
        data_b64 = payload["image"]["data_b64"]
    except Exception as e:
        return _json_error(
            400,
            "BAD_PAYLOAD",
            "Expected session_id, frame_id, ts_client_ms, image.data_b64",
            {"detail": str(e)},
        )

    try:
        if "base64," in data_b64:
            data_b64 = data_b64.split("base64,", 1)[1]
        raw = base64.b64decode(data_b64, validate=False)
        arr = np.frombuffer(raw, np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        return _json_error(400, "IMAGE_DECODE_FAILED", f"Image decode failed: {str(e)}")

    h, w = frame_bgr.shape[:2]

    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires monotonically increasing timestamps per stream.
        # We use the client-provided ts_client_ms which is already monotonic enough for a single page session.
        try:
            ts_ms = int(ts_client_ms)
        except Exception:
            ts_ms = int(time.time() * 1000)

        with _DETECTOR_LOCK:
            result = DETECTOR_INSTANCE.detect_for_video(mp_image, ts_ms)

        objects = to_contract_objects(result, w, h)

        if focus_label:
            fl = str(focus_label).strip().lower()
            objects = [o for o in objects if str(o["name"]).strip().lower() == fl]

        top = {"name": objects[0]["name"], "score": objects[0]["score"]} if objects else None

    except Exception as e:
        return _json_error(500, "DETECT_FAILED", f"Detection failed: {str(e)}")

    ts_server_ms = int(time.time() * 1000)
    latency_ms = int((time.time() - t0) * 1000)

    return JsonResponse({
        "ok": True,
        "session_id": session_id,
        "frame_id": frame_id,
        "ts_client_ms": ts_client_ms,
        "ts_server_ms": ts_server_ms,
        "latency_ms": latency_ms,
        "focus_label": focus_label,
        "telemetry": telemetry,
        "objects": objects,
        "top": top,
        "warnings": ["NO_OBJECTS"] if not objects else [],
        "error": None,
    })

@dataclass
class ApiError:
    code: str
    message: str


def _error(code: str, message: str, http: int = 400) -> JsonResponse:
    return JsonResponse({"ok": False, "error": asdict(ApiError(code, message))}, status=http)


def _decode_b64_jpeg_to_rgb(b64: str) -> np.ndarray:
    """
    Returns RGB uint8 image: HxWx3
    """
    if not b64:
        raise ValueError("Empty base64 image")

    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode JPEG")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _normalize_handedness(label: str) -> str:
    """
    Your previous code flipped Left/Right because of mirrored camera.
    In your frontend, you already flip for front camera *when capturing*.
    So in server, we should NOT flip again.

    If you still want flipping, do it conditionally using telemetry.camera == "user".
    """
    return label


def _extract_top(result: vision.GestureRecognizerResult) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (top_gesture, top_handedness) with keys: name, score
    """
    top_gesture = None
    top_hand = None

    # handedness: list per hand -> list of categories
    if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
        h = result.handedness[0][0]
        top_hand = {"name": _normalize_handedness(h.category_name), "score": float(h.score)}

    # gestures: list per hand -> list of gesture categories
    if result.gestures and len(result.gestures) > 0 and len(result.gestures[0]) > 0:
        g = result.gestures[0][0]
        top_gesture = {"name": g.category_name, "score": float(g.score)}

    return top_gesture, top_hand

@csrf_exempt
def hand_recognition(request):
    if request.method != "POST":
        return _err("METHOD_NOT_ALLOWED", "POST only", http=405)

    t0 = time.time()
    ts_server_ms = int(time.time() * 1000)

    try:
        body = request.body.decode("utf-8") if request.body else ""
        data = json.loads(body) if body else {}
    except Exception:
        return _err("BAD_JSON", "Request body must be valid JSON", http=400)

    session_id = data.get("session_id", "")
    frame_id = data.get("frame_id", None)

    image = data.get("image") or {}
    b64 = image.get("data_b64")

    if not b64:
        return _err("IMAGE_MISSING", "image.data_b64 missing", http=400)

    img_bgr = _decode_b64_jpeg(b64)
    if img_bgr is None:
        return _err("IMAGE_DECODE_FAILED", "Could not decode base64 jpeg", http=400)

    # Run model
    try:
        hands = hand_service.detect(img_bgr)
    except Exception as e:
        return _err("MODEL_ERROR", "Hand model failed", http=500, extra={"detail": str(e)})

    latency_ms = int((time.time() - t0) * 1000)

    return JsonResponse({
        "ok": True,
        "session_id": session_id,
        "frame_id": frame_id,
        "ts_server_ms": ts_server_ms,
        "latency_ms": latency_ms,
        "hands": hands,
        "warnings": [],
        "error": None
    })