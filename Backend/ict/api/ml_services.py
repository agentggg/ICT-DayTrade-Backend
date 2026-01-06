"""
Central registry for ALL heavyweight ML/CV models and services.

Goals:
- No heavy model loads at Django import-time
- Thread-safe singletons / cached instances
- One place to manage memory / performance
"""

from __future__ import annotations

import os
import time
import base64
import binascii
import threading
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp

from django.conf import settings

# Must be set BEFORE transformers/tokenizers get imported
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------
# NLP / Transformers / Embeddings (LAZY)
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_happy_tt():
    from happytransformer import HappyTextToText
    return HappyTextToText("T5", "vennify/t5-base-grammar-correction")


@lru_cache(maxsize=1)
def get_beam_settings():
    from happytransformer import TTSettings
    return TTSettings(min_length=1)


@lru_cache(maxsize=1)
def get_gec_pipeline():
    from transformers import pipeline
    return pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")


@lru_cache(maxsize=1)
def get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/stsb-roberta-base")


@lru_cache(maxsize=1)
def get_keybert():
    from keybert import KeyBERT
    model = get_sentence_transformer()
    return KeyBERT(model=model)


# ---------------------------------------------------------------------
# CV: YOLO (LAZY)
# ---------------------------------------------------------------------
_YOLO_LOCK = threading.Lock()
_YOLO_V8N = None
_YOLO_V8X = None


def get_yolo_v8n():
    global _YOLO_V8N
    if _YOLO_V8N is not None:
        return _YOLO_V8N
    with _YOLO_LOCK:
        if _YOLO_V8N is None:
            from ultralytics import YOLO
            _YOLO_V8N = YOLO("yolov8n.pt")
    return _YOLO_V8N


def get_yolo_v8x():
    global _YOLO_V8X
    if _YOLO_V8X is not None:
        return _YOLO_V8X
    with _YOLO_LOCK:
        if _YOLO_V8X is None:
            from ultralytics import YOLO
            _YOLO_V8X = YOLO("yolov8x.pt")
    return _YOLO_V8X


# ---------------------------------------------------------------------
# CV: Face Recognition (imports are heavy; keep them in functions)
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_face_recognition():
    import face_recognition
    return face_recognition


# ---------------------------------------------------------------------
# MediaPipe: shared paths
# ---------------------------------------------------------------------
MP_MODEL_DIR = Path(settings.BASE_DIR) / "model"
MP_OBJECT_MODEL_PATH = str(MP_MODEL_DIR / "efficientdet_lite2.tflite")
MP_GESTURE_MODEL_PATH = str(MP_MODEL_DIR / "gesture_recognizer.task")
MP_HAND_LANDMARKER_PATH = str(MP_MODEL_DIR / "hand_landmarker.task")


# ---------------------------------------------------------------------
# MediaPipe Object Detector (VIDEO mode singleton)
# ---------------------------------------------------------------------
_DETECTOR_LOCK = threading.Lock()
_DETECTOR_INSTANCE = None


def get_mp_object_detector_video():
    """
    Returns a cached MediaPipe ObjectDetector in VIDEO mode.
    NOTE: VIDEO mode requires monotonically increasing timestamps per stream.
    """
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is not None:
        return _DETECTOR_INSTANCE

    with _DETECTOR_LOCK:
        if _DETECTOR_INSTANCE is not None:
            return _DETECTOR_INSTANCE

        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        RunningMode = mp.tasks.vision.RunningMode
        BaseOptions = mp.tasks.BaseOptions

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=MP_OBJECT_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            max_results=5,
            score_threshold=0.30,
        )
        _DETECTOR_INSTANCE = ObjectDetector.create_from_options(options)
        return _DETECTOR_INSTANCE


# ---------------------------------------------------------------------
# MediaPipe Gesture Recognizer (IMAGE mode singleton)
# ---------------------------------------------------------------------
_GESTURE_LOCK = threading.Lock()
_GESTURE = None


def get_mp_gesture_recognizer_image():
    """
    Returns a cached MediaPipe GestureRecognizer in IMAGE mode.
    Recognizer is not guaranteed thread-safe, so lock during inference.
    """
    global _GESTURE
    if _GESTURE is not None:
        return _GESTURE

    with _GESTURE_LOCK:
        if _GESTURE is not None:
            return _GESTURE

        if not os.path.isfile(MP_GESTURE_MODEL_PATH):
            raise FileNotFoundError(f"Missing gesture model: {MP_GESTURE_MODEL_PATH}")

        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        RunningMode = mp.tasks.vision.RunningMode
        BaseOptions = mp.tasks.BaseOptions

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=MP_GESTURE_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
        )
        _GESTURE = GestureRecognizer.create_from_options(options)
        return _GESTURE


def mp_gesture_lock():
    """
    Expose the same lock to consumers so they can lock inference.
    """
    return _GESTURE_LOCK


# ---------------------------------------------------------------------
# HandLandmarkService import (robust)
# ---------------------------------------------------------------------
def import_hand_landmark_service():
    """
    Import HandLandmarkService robustly from common filenames.
    """
    try:
        from .hand_recognition import HandLandmarkService  # type: ignore
        return HandLandmarkService
    except Exception:
        try:
            from .hand_landmarker_service import HandLandmarkService  # type: ignore
            return HandLandmarkService
        except Exception:
            import importlib.util

            api_dir = Path(__file__).resolve().parent
            candidates = [
                api_dir / "hand_recognition.py",
                api_dir / "hand recognition.py",
                api_dir / "hand_landmarker_service.py",
                api_dir / "hand_landmarker.py",
            ]

            last_err = None
            for p in candidates:
                if not p.exists():
                    continue
                try:
                    spec = importlib.util.spec_from_file_location("ict.api.hand_landmark_fallback", str(p))
                    if spec is None or spec.loader is None:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore
                    return getattr(mod, "HandLandmarkService")
                except Exception as e:
                    last_err = e

            existing = [str(p.name) for p in candidates if p.exists()]
            hint = (
                "Could not import HandLandmarkService.\n"
                "Fix (recommended): rename your file to `hand_recognition.py` and keep it in `Backend/ict/api/`.\n"
                "Ensure `Backend/ict/api/__init__.py` exists.\n\n"
                f"Searched: {[p.name for p in candidates]}\n"
                f"Existing: {existing or 'NONE'}"
            )
            raise ImportError(hint + (f"\n\nLast loader error: {last_err}" if last_err else ""))


# ---------------------------------------------------------------------
# Helpers: base64 decoding (shared)
# ---------------------------------------------------------------------
def decode_b64_jpeg_to_bgr(b64: str) -> np.ndarray:
    """
    Decode base64 jpeg -> OpenCV BGR.
    Accepts raw base64 or data URLs.
    """
    if not b64 or not isinstance(b64, str):
        raise ValueError("image.data_b64 missing or invalid")

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


def bgr_to_mp_image(img_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)