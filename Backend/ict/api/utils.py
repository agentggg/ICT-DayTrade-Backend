def detect_fvg_from_array(c):
    """
    c must be 3 candles:
    c[0] → first
    c[1] → middle
    c[2] → last
    """

    # Unpack
    c0, c1, c2 = c

    # Validate structure
    for k in ("high", "low"):
        if k not in c0 or k not in c1 or k not in c2:
            raise Exception("Candles missing high/low values")

    # BULLISH FVG:
    # c0.high < c2.low AND middle candle has a gap
    if c0["high"] < c2["low"]:
        return {
            "type": "bullish",
            "gap_high": c2["low"],
            "gap_low": c0["high"],
        }

    # BEARISH FVG:
    # c0.low > c2.high
    if c0["low"] > c2["high"]:
        return {
            "type": "bearish",
            "gap_high": c0["low"],
            "gap_low": c2["high"],
        }

    return None

def is_bullish(c):
    return c["close"] > c["open"]

def is_bearish(c):
    return c["close"] < c["open"]

def body_ratio(c):
    body = abs(c["close"] - c["open"])
    wick_total = c["high"] - c["low"]
    if wick_total == 0:
        return 0
    return body / wick_total

def detect_displacement(prev, curr, direction):
    """
    Basic ICT displacement check:
    - Bullish: strong up candle breaking previous structure
    - Bearish: strong down candle breaking previous structure
    """
    if direction == "bullish":
        return curr["high"] > prev["high"] and is_bullish(curr)
    else:
        return curr["low"] < prev["low"] and is_bearish(curr)

def find_order_blocks(
    candles,
    direction="bullish",
    require_displacement=True,
    min_body_ratio=0.45,
    mss_lookback=3,
    ob_lookback=4,
    min_ob_body=0.25,
):
    """
    ICT-Style Order Block Identification

    Returns a list of:
    {
        "direction": "bullish" | "bearish",
        "ob": {
            "time": ...,
            "high": ...,
            "low": ...
        }
    }
    """

    matches = []

    n = len(candles)
    if n < 5:
        return []

    for i in range(2, n):  # start from 2 to allow lookbacks
        prev = candles[i - 1]
        curr = candles[i]

        # 1️⃣ Check displacement
        if require_displacement and not detect_displacement(prev, curr, direction):
            continue

        # 2️⃣ MSS check: look back mss_lookback candles
        if direction == "bullish":
            mss = any(candles[i]["high"] > candles[i - k]["high"]
                      for k in range(1, min(mss_lookback + 1, i)))
        else:
            mss = any(candles[i]["low"] < candles[i - k]["low"]
                      for k in range(1, min(mss_lookback + 1, i)))

        if not mss:
            continue

        # 3️⃣ Identify the OB candle (last opposite candle before displacement)
        start = max(0, i - ob_lookback)
        ob_candidates = candles[start:i]

        if direction == "bullish":
            # OB = last bearish candle before displacement
            pool = [c for c in ob_candidates if is_bearish(c)]
        else:
            # OB = last bullish candle before displacement
            pool = [c for c in ob_candidates if is_bullish(c)]

        if not pool:
            continue

        ob = pool[-1]  # last valid opposite candle (ICT rule)

        # 4️⃣ Ensure OB body strength
        if body_ratio(ob) < min_ob_body:
            continue

        matches.append({
            "direction": direction,
            "ob": {
                "time": ob["time"],
                "high": ob["high"],
                "low": ob["low"]
            }
        })

    return matches


    # ============================================================
# View helpers (keep views.py thin)
# ============================================================
# NOTE:
# - Imports are intentionally inside functions where possible to avoid heavy
#   import costs and to prevent circular import issues.

import threading
from typing import Dict

# MediaPipe GestureRecognizer is not guaranteed thread-safe.
# We source the shared lock from ml_services so all call sites coordinate.
try:
    from .ml_services import mp_gesture_lock as _MP_GESTURE_LOCK
except Exception:
    _MP_GESTURE_LOCK = threading.Lock()

GESTURE_LOCK = _MP_GESTURE_LOCK


def get_gesture_recognizer():
    """Return a cached MediaPipe GestureRecognizer (IMAGE mode)."""
    # Cached instance is owned by ml_services; we simply access it here.
    from .ml_services import get_mp_gesture_recognizer_image
    return get_mp_gesture_recognizer_image()


# -----------------------------
# YOLO model cache
# -----------------------------
_YOLO_X_MODEL = None
_YOLO_X_LOCK = threading.Lock()


def get_model():
    """Cached YOLO model loader (yolov8x.pt)."""
    global _YOLO_X_MODEL
    if _YOLO_X_MODEL is None:
        with _YOLO_X_LOCK:
            if _YOLO_X_MODEL is None:
                from ultralytics import YOLO
                _YOLO_X_MODEL = YOLO("yolov8x.pt")
    return _YOLO_X_MODEL


# -----------------------------
# Small numeric / detection helpers
# -----------------------------
def clamp(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def normalize_label(s: str) -> str:
    return (s or "").strip().lower()


def xyxy_to_norm_xyxy(xyxy, w, h) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h}


def iou(a: Dict[str, float], b: Dict[str, float]) -> float:
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


def nms(dets, iou_thresh: float = 0.50):
    """Non-max suppression for a list of det dicts with keys: box, score."""
    dets = sorted(dets, key=lambda d: d.get("score", 0.0), reverse=True)
    keep = []
    for d in dets:
        if any(iou(d["box"], k["box"]) >= iou_thresh for k in keep):
            continue
        keep.append(d)
    return keep