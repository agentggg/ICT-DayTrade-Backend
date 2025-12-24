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


# ------------------------------------------------------------------------------
# YOLO (object detection) - keep your existing singleton if you want
# ------------------------------------------------------------------------------
YOLO_MODEL = YOLO("yolov8n.pt")  # process-level singleton


# ------------------------------------------------------------------------------
# MediaPipe Object Detector (EfficientDet Lite) - single safe instance
# ------------------------------------------------------------------------------
_DETECTOR_LOCK = threading.Lock()

MODEL_PATH = "model/efficientdet_lite2.tflite"  # adjust if needed

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

_DETECTOR_OPTIONS = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    max_results=5,
    score_threshold=0.30,
)

# Create ONE instance. Do NOT create two.
DETECTOR_INSTANCE = ObjectDetector.create_from_options(_DETECTOR_OPTIONS)

# Monotonic clock baseline for VIDEO timestamps
START_TIME = time.monotonic()


# ------------------------------------------------------------------------------
# Helpers (base64 jpeg -> OpenCV image, and contract formatting)
# ------------------------------------------------------------------------------
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
def predict(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": {"code": "METHOD_NOT_ALLOWED", "message": "POST only"}}, status=405)

    if "image" not in request.FILES:
        return JsonResponse({"ok": False, "error": {"code": "IMAGE_MISSING", "message": "Missing file field 'image'."}}, status=400)

    conf = _clamp(request.POST.get("conf", "0.25"), 0.01, 0.99)
    iou_thresh = _clamp(request.POST.get("iou", "0.50"), 0.05, 0.95)
    max_det = max(1, min(_safe_int(request.POST.get("max_det", "50"), 50), 300))
    imgsz = max(320, min(_safe_int(request.POST.get("imgsz", "960"), 960), 1920))

    focus_label = request.POST.get("focus_label", None)
    focus_label_norm = _normalize_label(focus_label) if focus_label else None

    t0 = time.perf_counter()
    try:
        uploaded = request.FILES["image"].read()
        pil = Image.open(io.BytesIO(uploaded)).convert("RGB")
        frame = np.array(pil)
    except Exception as e:
        return JsonResponse({"ok": False, "error": {"code": "IMAGE_DECODE_FAILED", "message": str(e)}}, status=400)

    h, w = frame.shape[:2]

    model = get_model()
    try:
        results = model.predict(frame, conf=conf, iou=iou_thresh, max_det=max_det, imgsz=imgsz, verbose=False)
    except Exception as e:
        return JsonResponse({"ok": False, "error": {"code": "INFERENCE_FAILED", "message": str(e)}}, status=500)

    r = results[0]
    raw = []
    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            name = model.names.get(cls_id, str(cls_id))
            score = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            box_norm = _xyxy_to_norm_xyxy(xyxy, w, h)
            raw.append({"class_id": cls_id, "name": str(name), "score": score, "box": box_norm})

    nms_in = [{"score": d["score"], "box": d["box"], "d": d} for d in raw]
    nms_out = _nms(nms_in, iou_thresh=iou_thresh)
    objects = [x["d"] for x in nms_out]
    objects.sort(key=lambda d: d["score"], reverse=True)
    objects = objects[:max_det]

    focus = {"enabled": False}
    if focus_label_norm:
        matches = [o for o in objects if _normalize_label(o["name"]) == focus_label_norm]
        best = matches[0] if matches else None
        focus = {"enabled": True, "name": focus_label_norm, "found": bool(best), "best_score": float(best["score"]) if best else 0.0}

    top = {"name": objects[0]["name"], "score": float(objects[0]["score"])} if objects else None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return JsonResponse({
        "ok": True,
        "ts_server_ms": int(time.time() * 1000),
        "latency_ms": latency_ms,
        "image": {"width": int(w), "height": int(h)},
        "focus": focus,
        "top": top,
        "objects": objects,
        "warnings": ["NO_OBJECTS"] if not objects else [],
        "error": None
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


# ------------------------------------------------------------------------------
# Hand recognition (FIXED: you must initialize recognizer; placeholder for now)
# ------------------------------------------------------------------------------
@csrf_exempt
def hand_recognition(request):
    """
    You had `_lock` and `_recognizer` referenced but not defined.
    This endpoint will return a clean error until you add your MediaPipe gesture recognizer init.
    """
    return JsonResponse({
        "ok": False,
        "error": {
            "code": "NOT_CONFIGURED",
            "message": "Gesture recognizer is not initialized in this file. Add _recognizer and _lock, then implement."
        }
    }, status=501)


# ------------------------------------------------------------------------------
# Object detection endpoint (MediaPipe EfficientDet) - FIXED + SIMPLE
# ------------------------------------------------------------------------------
@csrf_exempt
def object_gesture(request):
    t0 = time.perf_counter()

    if request.method != "POST":
        return JsonResponse({"ok": False, "error": {"code": "METHOD", "message": "POST only"}}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": {"code": "JSON", "message": "Invalid JSON"}}, status=400)

    img = payload.get("image") or {}
    data_b64 = img.get("data_b64")
    if not data_b64:
        return JsonResponse({"ok": False, "error": {"code": "IMAGE_MISSING", "message": "image.data_b64 missing"}}, status=400)

    try:
        frame_bgr = decode_b64_jpeg_to_bgr(data_b64)
    except Exception as e:
        return JsonResponse({"ok": False, "error": {"code": "DECODE", "message": str(e)}}, status=400)

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int((time.monotonic() - START_TIME) * 1000)

    try:
        with _DETECTOR_LOCK:
            detection_result = DETECTOR_INSTANCE.detect_for_video(mp_image, timestamp_ms)
    except Exception as e:
        return JsonResponse({"ok": False, "error": {"code": "DETECT", "message": f"Detector error: {str(e)}"}}, status=500)

    objects = to_contract_objects(detection_result, w, h)
    top = {"name": objects[0]["name"], "score": objects[0]["score"]} if objects else None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return JsonResponse({
        "ok": True,
        "ts_server_ms": int(time.time() * 1000),
        "latency_ms": latency_ms,
        "objects": objects,
        "top": top,
        "warnings": ["NO_OBJECTS"] if not objects else [],
        "error": None
    })