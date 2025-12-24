# ict/api/gesture_recognizer_service.py
from __future__ import annotations

from typing import Any, Dict
import threading
import os

import cv2
import numpy as np
import mediapipe as mp


class GestureRecognizerService:
    """
    Wrapper around MediaPipe GestureRecognizer (.task) for server-side JPEG/BGR frames.
    Returns JSON-friendly dict:
      { "gestures": [{"name":..., "score":...}, ...],
        "handedness": [{"name":..., "score":...}, ...] }
    """

    def __init__(self, model_asset_path: str):
        if not os.path.isfile(model_asset_path):
            raise FileNotFoundError(f"Missing gesture model: {model_asset_path}")

        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=RunningMode.IMAGE,
        )

        self._recognizer = GestureRecognizer.create_from_options(options)
        self._lock = threading.Lock()

    def process_bgr(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        if img_bgr is None or img_bgr.size == 0:
            return {"gestures": [], "handedness": []}

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            r = self._recognizer.recognize(mp_image)

        gestures_out = []
        if getattr(r, "gestures", None) and r.gestures and len(r.gestures[0]) > 0:
            for g in r.gestures[0]:
                gestures_out.append({
                    "name": getattr(g, "category_name", None),
                    "score": float(getattr(g, "score", 0.0) or 0.0),
                })

        handed_out = []
        if getattr(r, "handedness", None) and r.handedness and len(r.handedness[0]) > 0:
            for h in r.handedness[0]:
                handed_out.append({
                    "name": getattr(h, "category_name", None),
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                })

        return {"gestures": gestures_out, "handedness": handed_out}