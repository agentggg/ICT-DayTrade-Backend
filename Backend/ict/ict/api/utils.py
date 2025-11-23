def _body_ratio(candle: dict) -> float:
    h = candle["high"]
    l = candle["low"]
    o = candle["open"]
    c = candle["close"]

    rng = h - l
    if rng == 0:
        return 0.0
    return abs(c - o) / rng


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