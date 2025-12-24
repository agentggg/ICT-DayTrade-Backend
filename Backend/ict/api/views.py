from datetime import datetime, date, timedelta
from collections import Counter
from django.db import transaction
import random
import json
import logging
import threading
import re
import pprint
import copy

from django.core.exceptions import ObjectDoesNotExist
from django.core.mail import send_mail
from django.db import IntegrityError, transaction as db_transaction
from django.db.models import F, Value, CharField
from django.db.models.functions import Concat
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from datetime import timezone
utc = timezone.utc
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response 
from rest_framework.views import APIView
from rest_framework.authtoken.views import ObtainAuthToken
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.http import HttpResponse
from django.conf import settings


 
from .models import *
from .serializers import *
from .utils import *
import requests
from django.utils.html import escape


import io
from PIL import Image
import numpy as np
import os
from django.utils.timezone import now
import face_recognition

from ultralytics import YOLO
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

_model = None
_model_lock = threading.Lock()

model = YOLO("yolov8n.pt") # Load model once (process-level singleton)
MODEL_PATH = "model/gesture_recognizer.task"
 
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)

_recognizer = GestureRecognizer.create_from_options(_options)
_lock = threading.Lock()

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

    # Serialize once
    data = CandleSerializers(candles, many=True).data

    # Create sliding 3-candle windows
    candle_set = []
    # pprint(f"==>> candle_set: {candle_set}")
    temp_array = []
    for i in data:
        if len(temp_array) == int(amount_of_candles):
            candle_set.append(copy.deepcopy(temp_array))
            temp_array.clear()
        else:
            temp_array.append(i)

    if not candle_set:
        return Response({"error": "Not enough candles"}, status=400)

    # choose random valid index
    random_index = random.randint(0, len(candle_set) - 1)

    selected = candle_set[random_index]

    return Response(selected)

@api_view(['POST'])
def check_fvg(request):
    candles = request.data.get("candles", request.data)
    print(f"==>> candles: {candles}")

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
    
# views.py
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
        # you can tune these:
        mss_lookback=3,
        ob_lookback=4,
        min_ob_body=0.25,
    )

    if not matches:
        return Response({"ok": False, "matches": [], "reason": "no order block found"})

    return Response({"ok": True, "matches": matches})

@api_view(["POST"])
def check_manual_ob(request):
    """
    Body:
      {
        "candles": [ {time, open, high, low, close}, ... ]   // exactly 7
      }

    Response:
      {
        "ok": bool,
        "is_ob": bool,
        "direction": "bullish"|"bearish"|null,
        "prev_direction": "bullish"|"bearish"|"none",
        "reason": "text"
      }
    """
    candles = request.data.get("candles", [])

    if not isinstance(candles, list) or len(candles) < 5:
        return Response(
            {
                "ok": False,
                "is_ob": False,
                "direction": None,
                "prev_direction": "none",
                "reason": "need at least 5 candles (expected ~7)",
            }
        )

    result = confirm_manual_ob(candles)
    return Response(result)  

@api_view(["POST"])
def check_breaker_block(request):
    direction = request.data.get("direction")  # "bullish" or "bearish"
    candles = request.data.get("candles", [])

    if not isinstance(candles, list) or len(candles) < 5:
        return Response({"ok": False, "reason": "need at least 5 candles"}, status=400)

    # You said you'll send 12; just in case, we only look at the last 12
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
        # Initialize your serializer with the request data
        serializer = ObtainAuthToken.serializer_class(data=request.data, context={'request': request})

        # Validate the data. If it's not valid, a ValidationError will be raised
        serializer.is_valid(raise_exception=True)

        # Get the user from the validated data
        user = serializer.validated_data['user']

        # Fetch profile access roles. This assumes the user model has a related 'profile_access' field
        profile_access_roles = user.profile_access.all().values_list('name', flat=True)

        # Attempt to get or create the auth token for the user
        token, created = Token.objects.get_or_create(user=user)

        # If everything is successful, return the user information and token
        return Response({
            'token': token.key,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'username': user.username,
            'email': user.email,
            'active': user.is_active,
            'profile_access': list(profile_access_roles)
        })

    # except ValidationError as e:
    #     print(e)
    #     # Handle validation errors from serializer.is_valid()
    #     return Response({"errors": e.detail}, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        # Handle any other exceptions that may occur
        print(e)
        return Response({"errors": "An unexpected error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except User.DoesNotExist:
        # Handle case where the user does not exist
        return Response({"error": "User does not exist"}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        # Catch any other unexpected errors
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["GET", "POST"])
def trade_journal_view(request):
    """
    POST /api/trades/
        - Create/save a trade
           
    GET /api/trades/
        - Get all trades
    
    GET /api/trades/?trade_id=1
        - Get a single trade by internal ID (pk)

    GET /api/trades/?external_trade_id=1692251887
        - Get a single trade by external_trade_id
    """

    # CREATE
    if request.method == "POST":
        serializer = TradeJournalSerializer(data=request.data)
        if serializer.is_valid():            
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 
    # READ
    trade_id = request.query_params.get("trade_id")
    external_trade_id = request.query_params.get("external_trade_id")

    # Get by internal trade id (pk)
    if trade_id is not None:
        trade = get_object_or_404(TradeJournal, pk=trade_id)
        serializer = TradeJournalSerializer(trade)
        return Response(serializer.data, status=status.HTTP_200_OK)

    # Get by external trade id
    if external_trade_id is not None:
        trade = get_object_or_404(TradeJournal, external_trade_id=external_trade_id)
        serializer = TradeJournalSerializer(trade)
        return Response(serializer.data, status=status.HTTP_200_OK)


    # Get all
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
    # CREATE
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
    """
        { 
        "date": "2025-11-25",
        "time": "09:59",
        "symbol": "MNQ",
        "direction": "long",
        "session": "newyork",
        "timeframe": "1m",
        "risk_percent": 0.5,
        "entry_price": 24816.75,
        "exit_price": 24855.25,
        "result": "win",
        "pnl": 77.0,
        "r_multiple": 3.0,
        "ict_setup": "IFVG reversal after MSS",
        "higher_tf_bias": "4H",
        "ifvg_used": true,
        "ob_used": false,
        "breaker_used": false,
        "liquidity_sweep_used": true,
        "mitigation_block_used": false,
        "pre_trade_emotion": "calm",
        "post_trade_emotion": "satisfied",
        "trade_grade": "A",
        "followed_plan": true,
        "emotional_trade": false,
        "took_profit_early": false,
        "missed_trade": false,
        "moved_stop_loss": false,
        "revenge_trade": false,
        "notes": "TopstepX copy output: Trade ID 1692251887, size 1 contract(s) on MNQ, entered around \"November 25 2025 @ 9:59:53 am\", exited around \"November 25 2025 @ 11:45:17 am\", entry price 24,816.75, exit price 24,855.25, realized PnL $77.00, commissions $0, extra metric $-0.74.",
        "stop_levels": [
            {
            "price": 24800.0,
            "reason": "Initial SL below 1m OB low."
            },
            {
            "price": 24820.0,
            "reason": "Moved SL to BE+ after price reached 1R."
            }
        ],
        "take_profit_levels": [
            {
            "price": 24860.0,
            "reason": "First partial at intraday swing high."
            },
            {
            "price": 24900.0,
            "reason": "Final TP at HTF premium level."
            }
        ]
        }

        1692251887
        /MNQ
        1
        November 25 2025 @ 9:59:53 am
        November 25 2025 @ 11:45:17 am
        01:45:23
        24,816.75
        24,855.25
        $77.00
        $0
        $-0.74
        Long

            {
        "date": "2025-11-26",
        "time": "09:30",
        "symbol": "MNQ",
        "timeframe": "1m",
        "ict_setup": "Breaker Block Model (BB)",
        "session": "newyork",
        "outcome": "failed",
        "what_happened": "BB formed after sweep but price failed to respect the block and traded through.",
        "why_outcome": "I forced a BB in counter HTF direction; HTF liquidity was actually above, so the move continued higher.",
        "notes": "Need to require MSS on 5m in direction of BB before taking 1m entries.",
        "strategy_modification": true,
        "modification_details": "Only trade BB if 5m market structure and liquidity profile confirms direction and there's a clear external range.",
        "screenshot_link": "https://www.tradingview.com/x/your-chart-id/"
        }
    """


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO("yolov8x.pt")  # or yolov8s.pt, etc.
    return _model

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


class FacialRecognitionTool:
    def __init__(self):
        # Store known_faces under a predictable, server-safe directory
        self.KNOWN_FACES_DIR = os.path.join(settings.BASE_DIR, "known_faces")

        self.TOLERANCE = 0.50      # maximum allowed distance; higher means "different person"
        self.FRAME_SCALE = 0.50    # scale frame to 25% for speed

    def _decode_uploaded_image(self, uploaded_file):
        """
        UploadedFile -> BGR np.ndarray (OpenCV).
        Safe even if file was read previously.
        """
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        data = uploaded_file.read()
        if not data:
            return None

        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img_bgr

    def ensure_dir(self, path: str):
        """
        If the folder exists, do nothing; if not, create it (no crash).
        Equivalent to: mkdir -p path

        Args:
            path (str): the folder path to create if missing
        """
        os.makedirs(path, exist_ok=True)

    def load_facial_database(self):
        """
        Loads the facial entries into memory for faster processing.
        known_names lines up with the face encodings (index-aligned lists).

        If the folder does not exist, create it and return an empty list.
        This allows the program to run without errors.
        If the database is empty, recognition will return "Unknown".
        One folder is one person.

        We iterate through each enrolled person:
          - build the full path
          - skip anything that isn't a folder

        encodings.npy is the file inside each person folder.
        If it does not exist, we skip that folder.

        Returns:
            (known_encodings, known_names)
        """
        known_encodings = []
        known_names = []

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

        except Exception as e:
            print(f"[DB] Error loading database: {e}")
            return [], []

        print(f"[DB] Loaded {len(known_encodings)} encodings for {len(set(known_names))} people.")
        return known_encodings, known_names

    def save_encoding(self, name: str, encoding: np.ndarray):
        """
        Takes one face embedding and permanently attaches it to one person.

        Creates folder: known_faces/<name>/
        Appends to: encodings.npy (shape grows from (N,128) to (N+1,128))

        Args:
            name (str): name of the person
            encoding (np.ndarray): embedding of the face characteristics
        """
        person_dir = os.path.join(self.KNOWN_FACES_DIR, name)
        self.ensure_dir(person_dir)

        enc_path = os.path.join(person_dir, "encodings.npy")

        if os.path.exists(enc_path):
            encs = np.load(enc_path)
            encs = np.vstack([encs, encoding])
        else:
            encs = np.array([encoding])

        np.save(enc_path, encs)
        print(f"[ENROLL] Saved encoding for {name}. Total samples: {encs.shape[0]}")
        return int(encs.shape[0])

    def best_match(self, known_encodings, known_names, face_encoding):
        """
        Compares a single face encoding to all known encodings using face distance.

        Returns:
            (name, dist)
        """
        if not known_encodings:
            return "Unknown", 999.0

        dists = face_recognition.face_distance(known_encodings, face_encoding)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])

        if dist <= self.TOLERANCE:
            return known_names[idx], dist
        return "Unknown", dist

    def _decode_uploaded_image(self, uploaded_file):
        """
        Convert Django UploadedFile -> OpenCV BGR image (np.ndarray).
        """
        data = uploaded_file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img_bgr

    def recognize_from_upload(self, uploaded_file):
        frame = self._decode_uploaded_image(uploaded_file)
        if frame is None:
            return {"recognized": False, "name": "Unknown", "distance": 999.0, "error": "Invalid image"}

        # (optional) temporarily save what the backend sees for debugging
        # cv2.imwrite("/tmp/tektos_last.jpg", frame)

        # Scale down (but not too much)
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

        # IMPORTANT: do NOT return the numpy encoding in the response
        return {"recognized": name != "Unknown", "name": name, "distance": dist, "_encoding": chosen_enc}

    def enroll_from_upload(self, uploaded_file, name: str):
        result = self.recognize_from_upload(uploaded_file)
        if "error" in result:
            return {"ok": False, "error": result["error"]}

        enc = result.get("_encoding")
        if enc is None:
            return {"ok": False, "error": "No face detected"}

        samples_total = self.save_encoding(name, enc)
        return {"ok": True, "enrolled": True, "name": name, "samples_total": samples_total}
tool = FacialRecognitionTool()


@csrf_exempt
def recognize(request):
    """
    recognize endpoint

    request:
      Content-Type: multipart/form-data
      image: <file>

    response:
      {
        "ok": true,
        "recognized": true|false,
        "name": "Stevenson"|"Unknown",
        "distance": 0.42,
        "timestamp": "2025-12-22T20:15:00-05:00"
      }
    """
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
    """
    enroll endpoint

    request:
      Content-Type: multipart/form-data
      image: <file>
      name: <string>

    response:
      {
        "ok": true,
        "enrolled": true,
        "name": "Stevenson",
        "samples_total": 5,
        "timestamp": "2025-12-22T20:16:02-05:00"
      }
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=405)

    uploaded = request.FILES.get("image")
    name = (request.POST.get("name") or "").strip()

    if not uploaded:
        return JsonResponse({"ok": False, "error": "Missing form-data field: image"}, status=400)
    if not name:
        return JsonResponse({"ok": False, "error": "Missing form-data field: name"}, status=400)

    # Optional: basic filesystem-safe name rule (recommended)
    # This does NOT change your recognition logic; it avoids bad folder names.
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


# IMPORTANT: single shared recognizer must be locked (thread safety)


def _strip_data_url_prefix(b64: str) -> str:
    if not b64:
        return b64
    if "base64," in b64:
        return b64.split("base64,", 1)[1]
    return b64
 
def _decode_jpeg_b64_to_bgr(b64: str) -> np.ndarray:
    """
    Decodes base64 JPEG -> OpenCV BGR image.
    """
    b64 = _strip_data_url_prefix(b64)
    import base64
    raw = base64.b64decode(b64, validate=False)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if bgr is None:
        raise ValueError("cv2.imdecode returned None (invalid jpeg?)")
    return bgr

def _normalize_result(result) -> dict:
    """
    Convert MediaPipe result to frontend contract.
    """
    top_gesture = None
    gestures_out = []
    handedness_out = []

    if result and result.gestures and len(result.gestures) > 0 and len(result.gestures[0]) > 0:
        # first hand's gesture list
        for g in result.gestures[0][:5]:
            gestures_out.append({"name": g.category_name, "score": float(g.score)})

        best = result.gestures[0][0]
        top_gesture = {"name": best.category_name, "score": float(best.score)}

    if result and result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
        h = result.handedness[0][0]
        handedness_name = h.category_name
        handedness_score = float(h.score)

        # NOTE:
        # If your frontend is mirrored (front camera), handedness may appear flipped visually.
        # Do NOT flip here unless you are 100% sure you want "screen-left/right" semantics.
        handedness_out.append({"name": handedness_name, "score": handedness_score})

    return {
        "top_gesture": top_gesture,
        "gestures": gestures_out,
        "handedness": handedness_out,
    }

@csrf_exempt
def hand_recognition(request):
    import time
    """
    Django endpoint that uses your OpenCV->RGB->mp.Image->recognize_for_video strategy.
    """
    t0 = time.time()

    if request.method != "POST":
        return JsonResponse({"ok": False, "error": {"code": "METHOD", "message": "POST required"}}, status=405)

    # Parse JSON
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": {"code": "JSON", "message": "Invalid JSON"}}, status=400)

    session_id = payload.get("session_id")
    frame_id = payload.get("frame_id")
    ts_client_ms = payload.get("ts_client_ms")

    image_obj = payload.get("image") or {}
    data_b64 = image_obj.get("data_b64")

    if not data_b64:
        return JsonResponse({"ok": False, "error": {"code": "IMAGE", "message": "image.data_b64 missing"}}, status=400)

    # Decode JPEG -> OpenCV BGR
    try:
        frame_bgr = _decode_jpeg_b64_to_bgr(data_b64)
    except Exception as e:
        return JsonResponse({"ok": False, "error": {"code": "DECODE", "message": str(e)}}, status=400)

    # Match your desktop pipeline:
    # (optional) flip if your frontend is mirrored. Generally, the frontend already mirrors.
    # frame_bgr = cv2.flip(frame_bgr, 1)

    # Convert BGR -> RGB for MediaPipe
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # mp.Image from RGB numpy
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # VIDEO mode requires increasing timestamps (ms).
    # Use client ts if provided; else server now.
    ts_ms = int(ts_client_ms) if ts_client_ms else int(time.time() * 1000)

    # Recognize (one call)
    try:
        with _lock:
            result = _recognizer.recognize_for_video(mp_image, ts_ms)
    except Exception as e:
        return JsonResponse({"ok": False, "error": {"code": "INFER", "message": str(e)}}, status=500)

    latency_ms = int((time.time() - t0) * 1000)
    normalized = _normalize_result(result)

    warnings = []
    if not normalized["top_gesture"]:
        warnings.append("NO_HAND_OR_GESTURE")

    return JsonResponse({
        "ok": True,
        "session_id": session_id,
        "frame_id": frame_id,
        "ts_server_ms": int(time.time() * 1000),
        "latency_ms": latency_ms,
        "result": normalized,
        "warnings": warnings,
        "error": None
    })