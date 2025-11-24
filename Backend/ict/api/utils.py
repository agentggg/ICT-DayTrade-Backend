def _body_ratio(candle: dict) -> float:
    h = candle["high"]
    l = candle["low"]
    o = candle["open"]
    c = candle["close"]

    rng = h - l
    if rng == 0:
        return 0.0
    return abs(c - o) / rng

def find_order_blocks(
    candles: list,
    direction: str = "bullish",
    require_displacement: bool = True,
    min_body_ratio: float = 0.20,
) -> list:
    """
    Scan sequential pairs in `candles` and return list of matches:
      [{ "ob_index": i, "trigger_index": i+1, "ob": c0, "trigger": c1, "direction": "..."}]
    direction: "bullish" or "bearish"
    """
    matches = []
    n = len(candles)
    if n < 2:
        return matches

    for i in range(n - 1):
        c0 = candles[i]
        c1 = candles[i + 1]

        if direction == "bullish":
            ok = is_bullish_order_block(
                c0, c1, require_displacement=require_displacement, min_body_ratio=min_body_ratio
            )
        elif direction == "bearish":
            ok = is_bearish_order_block(
                c0, c1, require_displacement=require_displacement, min_body_ratio=min_body_ratio
            )
        else:
            ok = False

        if ok:
            matches.append(
                {
                    "ob_index": i,
                    "trigger_index": i + 1,
                    "ob": c0,
                    "trigger": c1,
                    "direction": direction,
                }
            )

    return matches

def is_bullish_order_block(
    c0: dict,
    c1: dict,
    require_displacement: bool = True,
    min_body_ratio: float = 0.20,   # <--- LOWER, works with your examples
) -> bool:
    """
    Bullish Order Block:
      - c0 = bearish candle (last down)
      - c1 = bullish candle breaking above c0 high
      - optional displacement body filter
    """

    # c0 must be bearish
    if not (c0["close"] < c0["open"]):
        return False

    # c1 must be bullish
    if not (c1["close"] > c1["open"]):
        return False

    # wick break of c0 high (ICT style)
    if not (c1["high"] > c0["high"]):
        return False

    # displacement body requirement (optional)
    if require_displacement and _body_ratio(c1) < min_body_ratio:
        return False

    return True


def is_bearish_order_block(
    c0: dict,
    c1: dict,
    require_displacement: bool = True,
    min_body_ratio: float = 0.20,
) -> bool:
    """
    Bearish Order Block:
      - c0 = bullish candle (last up)
      - c1 = bearish candle breaking below c0 low
      - optional displacement body filter
    """

    # c0 must be bullish
    if not (c0["close"] > c0["open"]):
        return False

    # c1 must be bearish
    if not (c1["close"] < c1["open"]):
        return False

    # wick break of c0 low
    if not (c1["low"] < c0["low"]):
        return False

    # displacement body requirement (optional)
    if require_displacement and _body_ratio(c1) < min_body_ratio:
        return False

    return True


def detect_fvg_from_array(candles, tolerance=0.0):
    """
    candles: list of exactly 3 candle dicts from backend
    tolerance: small allowed overlap (optional)

    Returns:
        None  -> no FVG
        dict  -> { type: 'bullish'/'bearish', gap_high: float, gap_low: float }
    """
    if len(candles) != 3:
        raise ValueError("FVG detection requires exactly 3 candles.")

    c1, c2, c3 = candles

    h1 = float(c1["high"])
    l1 = float(c1["low"])
    h3 = float(c3["high"])
    l3 = float(c3["low"])

    # Bullish FVG: candle1.high < candle3.low
    bullish = h1 < (l3 - tolerance)

    # Bearish FVG: candle1.low > candle3.high
    bearish = l1 > (h3 + tolerance)

    if bullish:
        return {
            "type": "bullish",
            "gap_high": h1,
            "gap_low": l3,
        }

    if bearish:
        return {
            "type": "bearish",
            "gap_high": l1,
            "gap_low": h3,
        }

    return None

def _body_ratio(candle: dict) -> float:
    h = candle["high"]
    l = candle["low"]
    o = candle["open"]
    c = candle["close"]

    rng = h - l
    if rng == 0:
        return 0.0
    return abs(c - o) / rng

from typing import List, Tuple


def is_bullish_breaker(
    candles: List[dict],
    min_ob_body: float = 0.35,
    min_breaker_body: float = 0.35,
) -> Tuple[bool, dict]:
    """
    Return (True/False, details) for a Bullish Breaker Block using the last N candles.

    Pattern:
      - Find a bullish OB.
      - Find a later strong bearish candle that closes below that OB low (failed OB).
      - Last candle must tag breaker body and close bullish.
    """
    n = len(candles)
    if n < 5:
        return False, {"reason": "need at least 5 candles"}

    # 1) Find the *last* bullish OB inside the window
    ob_index = None
    for i in range(0, n - 2):
        c0 = candles[i]
        c1 = candles[i + 1]
        if is_bullish_order_block(c0, c1, min_body_ratio=min_ob_body):
            ob_index = i

    if ob_index is None:
        return False, {"reason": "no bullish OB found"}

    ob = candles[ob_index]

    # 2) After OB, find a strong bearish candle that *fails* the OB
    breaker_index = None
    for j in range(ob_index + 2, n - 1):  # leave the last candle for the retest
        cj = candles[j]
        if cj["close"] < cj["open"] and _body_ratio(cj) >= min_breaker_body:
            # close below the OB low = OB failed
            if cj["close"] < ob["low"]:
                breaker_index = j

    if breaker_index is None:
        return False, {"reason": "no OB failure / breaker candle"}

    breaker = candles[breaker_index]

    # 3) Last candle = retest + bullish reaction
    last = candles[-1]

    # touch breaker body (between breaker.open and breaker.close)
    br_low = min(breaker["open"], breaker["close"])
    br_high = max(breaker["open"], breaker["close"])

    touched = (last["low"] <= br_high) and (last["high"] >= br_low)
    bullish_reaction = last["close"] > last["open"]

    if not touched:
        return False, {"reason": "last candle did not retest breaker body"}

    if not bullish_reaction:
        return False, {"reason": "no bullish reaction at breaker"}

    return True, {
        "reason": "bullish breaker",
        "ob_index": ob_index,
        "breaker_index": breaker_index,
    }

def is_bearish_breaker(
    candles: List[dict],
    min_ob_body: float = 0.35,
    min_breaker_body: float = 0.35,
) -> Tuple[bool, dict]:
    n = len(candles)
    if n < 5:
        return False, {"reason": "need at least 5 candles"}

    # 1) Find the last *bearish* OB (for shorts)
    ob_index = None
    for i in range(0, n - 2):
        c0 = candles[i]
        c1 = candles[i + 1]
        if is_bearish_order_block(c0, c1, min_body_ratio=min_ob_body):
            ob_index = i

    if ob_index is None:
        return False, {"reason": "no bearish OB found"}

    ob = candles[ob_index]

    # 2) Breaker candle: strong bullish close above ob.high
    breaker_index = None
    for j in range(ob_index + 2, n - 1):
        cj = candles[j]
        if cj["close"] > cj["open"] and _body_ratio(cj) >= min_breaker_body:
            if cj["close"] > ob["high"]:
                breaker_index = j

    if breaker_index is None:
        return False, {"reason": "no OB failure / breaker candle"}

    breaker = candles[breaker_index]

    # 3) Last candle retest + bearish reaction
    last = candles[-1]

    br_low = min(breaker["open"], breaker["close"])
    br_high = max(breaker["open"], breaker["close"])

    touched = (last["low"] <= br_high) and (last["high"] >= br_low)
    bearish_reaction = last["close"] < last["open"]

    if not touched:
        return False, {"reason": "last candle did not retest breaker body"}

    if not bearish_reaction:
        return False, {"reason": "no bearish reaction at breaker"}

    return True, {
        "reason": "bearish breaker",
        "ob_index": ob_index,
        "breaker_index": breaker_index,
    }


