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
from django.views.decorators.csrf import csrf_exempt


 
from .models import *
from .serializers import *
from .utils import *
import requests
from django.utils.html import escape

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
    """
    Expect body like:
    [
      { "id": ..., "_open": ..., "high": ..., "low": ..., "close": ..., ... },
      { ... },
      { ... }
    ]

    or:
    { "candles": [ {...}, {...}, {...} ] }
    """
    candles = request.data.get("candles", request.data)

    if not isinstance(candles, list):
        return Response(
            {"detail": "Expected a list of 3 candles or {'candles': [...]}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(candles) != 3:
        return Response(
            {"detail": "FVG check requires exactly 3 candles."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        result = detect_fvg_from_array(candles)
    except Exception as e:
        return Response(
            {"detail": f"Error while checking FVG: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if result is None:
        return Response(
            {
                "is_fvg": False,
                "type": None,
                "gap_high": None,
                "gap_low": None,
            }
        )

    return Response(
        {
            "is_fvg": True,
            "type": result["type"],       # 'bullish' or 'bearish'
            "gap_high": result["gap_high"],
            "gap_low": result["gap_low"],
        }
    )

@api_view(["POST"])
def check_order_block(request):
    """
    Accepts a list of candles (e.g. 13) and returns any detected order blocks.
    Request payload:
      { "direction": "bullish"|"bearish", "candles": [...], "require_displacement": true, "min_body_ratio": 0.2 }
    Response:
      { "ok": bool, "matches": [ {ob_index, trigger_index, ob, trigger, direction}, ... ], "reason": optional }
    """
    direction = request.data.get("direction")
    candles = request.data.get("candles", [])
    require_displacement = request.data.get("require_displacement", True)
    min_body_ratio = request.data.get("min_body_ratio", 0.20)

    if not isinstance(candles, list) or len(candles) < 2:
        return Response({"ok": False, "matches": [], "reason": "need at least 2 candles"})

    if direction not in ("bullish", "bearish"):
        return Response({"ok": False, "matches": [], "reason": "invalid direction"})

    matches = find_order_blocks(
        candles,
        direction=direction,
        require_displacement=require_displacement,
        min_body_ratio=min_body_ratio,
    )

    if not matches:
        return Response({"ok": False, "matches": [], "reason": "no order block found"})

    return Response({"ok": True, "matches": matches})

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

