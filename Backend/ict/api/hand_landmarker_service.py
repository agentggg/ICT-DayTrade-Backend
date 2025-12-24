from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

# MediaPipe has two common Python APIs:
# - mp.solutions (classic)  -> provides mp.solutions.hands
# - mp.tasks (newer)        -> provides HandLandmarker, requires a .task model asset
#
# Your environment currently raises:
#   AttributeError: module 'mediapipe' has no attribute 'solutions'
# which usually means either:
#   - a non-standard/incorrect mediapipe package was installed, OR
#   - a local file/module named `mediapipe.py` is shadowing the real package.
#
# This module supports BOTH APIs:
# - Prefer mp.solutions.hands if available.
# - Fallback to mp.tasks.vision.HandLandmarker if solutions is unavailable.

try:
    import mediapipe as mp  # type: ignore
except Exception as e:  # pragma: no cover
    mp = None  # type: ignore
    _mp_import_error = e
else:
    _mp_import_error = None


class HandLandmarkService:
    """
    Stateless detector wrapper around MediaPipe mp.solutions.hands.
    Input: BGR numpy image
    Output: list of hands in your contract format:
      [
        {
          "handedness": {"name":"Left|Right","score":0.98},
          "landmarks":[{"id":0,"x":...,"y":...,"z":...}, ... {"id":20,...}],
          "fingertips":[{"name":"THUMB","id":4,"x":...,"y":...}, ...]
        }
      ]
    """
    FINGERTIPS = [
        ("THUMB", 4),
        ("INDEX", 8),
        ("MIDDLE", 12),
        ("RING", 16),
        ("PINKY", 20),
    ]

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        # Only used if mp.solutions is NOT available (mp.tasks fallback):
        model_asset_path: Optional[str] = None,
    ):
        if mp is None:
            raise RuntimeError(
                "MediaPipe import failed. Ensure the official 'mediapipe' package is installed. "
                f"Import error: {_mp_import_error}"
            )

        self._use_solutions = bool(getattr(mp, "solutions", None) and getattr(mp.solutions, "hands", None))
        self._hands = None
        self._landmarker = None
        self._landmarker_close = None

        if self._use_solutions:
            # Classic API
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        # Fallback: Tasks API (requires a model asset .task file)
        # MediaPipe provides official task models; you must supply one.
        # Example model name: hand_landmarker.task
        try:
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore
        except Exception as e:
            # Provide a highly actionable error.
            mp_path = getattr(mp, "__file__", "<unknown>")
            raise RuntimeError(
                "Your installed 'mediapipe' package does not expose mp.solutions, and mp.tasks could not be imported. "
                "This often happens when a wrong package is installed or a local module shadows mediapipe. "
                f"Imported mediapipe from: {mp_path}. "
                "Fix steps: (1) ensure no local file/folder named 'mediapipe' in your project, "
                "(2) reinstall official mediapipe: pip uninstall -y mediapipe && pip install mediapipe. "
                f"Underlying error: {e}"
            )

        if not model_asset_path:
            mp_path = getattr(mp, "__file__", "<unknown>")
            raise RuntimeError(
                "MediaPipe 'solutions' is unavailable in this environment, so HandLandmarkService must use mp.tasks. "
                "To use mp.tasks you must provide model_asset_path (e.g., 'hand_landmarker.task'). "
                f"Imported mediapipe from: {mp_path}. "
                "If you expected mp.solutions.hands, you likely have a shadowed/incorrect mediapipe install."
            )

        base = mp_python.BaseOptions(model_asset_path=model_asset_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        # HandLandmarker has a close() method; keep reference for safe cleanup.
        self._landmarker_close = getattr(self._landmarker, "close", None)

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        if img_bgr is None or img_bgr.size == 0:
            return []

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        hands_out: List[Dict] = []

        if self._use_solutions:
            # Classic mp.solutions path
            result = self._hands.process(img_rgb)  # type: ignore[union-attr]
            if not result.multi_hand_landmarks:
                return []

            handedness_list = result.multi_handedness or []

            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # handedness
                name = "Unknown"
                score = 0.0
                if i < len(handedness_list) and handedness_list[i].classification:
                    cls = handedness_list[i].classification[0]
                    name = cls.label  # "Left" or "Right"
                    score = float(cls.score)

                # landmarks
                landmarks_out = []
                for lid, lm in enumerate(hand_landmarks.landmark):
                    landmarks_out.append({
                        "id": int(lid),
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    })

                # fingertips
                tips_out = []
                for tip_name, tip_id in self.FINGERTIPS:
                    lm = hand_landmarks.landmark[tip_id]
                    tips_out.append({
                        "name": tip_name,
                        "id": int(tip_id),
                        "x": float(lm.x),
                        "y": float(lm.y),
                    })

                hands_out.append({
                    "handedness": {"name": name, "score": score},
                    "landmarks": landmarks_out,
                    "fingertips": tips_out,
                })

            return hands_out

        # mp.tasks fallback
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)  # type: ignore[attr-defined]
        task_result = self._landmarker.detect(mp_image)  # type: ignore[union-attr]

        lm_list = getattr(task_result, "hand_landmarks", None) or []
        handed_list = getattr(task_result, "handedness", None) or []

        if not lm_list:
            return []

        for i, hand_landmarks in enumerate(lm_list):
            # handedness
            name = "Unknown"
            score = 0.0
            if i < len(handed_list) and handed_list[i]:
                cls = handed_list[i][0]
                name = getattr(cls, "category_name", None) or getattr(cls, "display_name", None) or "Unknown"
                score = float(getattr(cls, "score", 0.0))

            landmarks_out = []
            # task landmarks are NormalizedLandmark objects with x,y,z
            for lid, lm in enumerate(hand_landmarks):
                landmarks_out.append({
                    "id": int(lid),
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(getattr(lm, "z", 0.0)),
                })

            tips_out = []
            for tip_name, tip_id in self.FINGERTIPS:
                lm = hand_landmarks[tip_id]
                tips_out.append({
                    "name": tip_name,
                    "id": int(tip_id),
                    "x": float(lm.x),
                    "y": float(lm.y),
                })

            hands_out.append({
                "handedness": {"name": name, "score": score},
                "landmarks": landmarks_out,
                "fingertips": tips_out,
            })

        return hands_out


    def close(self) -> None:
        """Free underlying MediaPipe resources (safe to call multiple times)."""
        try:
            if self._hands is not None:
                # mp.solutions Hands implements close()
                close_fn = getattr(self._hands, "close", None)
                if callable(close_fn):
                    close_fn()
        finally:
            if self._landmarker_close and callable(self._landmarker_close):
                try:
                    self._landmarker_close()
                except Exception:
                    pass