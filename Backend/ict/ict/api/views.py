from datetime import datetime, date, timedelta
from collections import Counter
from django.db import transaction
import random
import json
import logging
import threading
import re

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
def fvg_data(request): 
    instrument = request.GET.get('instrument')
    timeframe = request.GET.get('timeframe')

    instrument_obj = Instrument.objects.get(name=instrument)

    candles = Candle.objects.filter(
        instrument=instrument_obj,
        timeframe=timeframe
    )

    # Serialize once
    data = CandleSerializers(candles, many=True).data

    # Create sliding 3-candle windows
    three_sets = []
    for i in range(len(data) - 2):
        c1 = data[i]
        c2 = data[i+1]
        c3 = data[i+2]
        three_sets.append([c1, c2, c3])

    if not three_sets:
        return Response({"error": "Not enough candles"}, status=400)

    # choose random valid index
    random_index = random.randint(0, len(three_sets) - 1)

    selected = three_sets[random_index]

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

@api_view(['POST'])
def check_order_block(request):
    direction = request.data.get("direction")  # "bullish" or "bearish"
    candles = request.data.get("candles", [])
    require_displacement = request.data.get("require_displacement", True)

    if len(candles) < 2:
        return Response({"ok": False, "reason": "need at least 2 candles"})

    # last two candles only
    c0 = candles[-2]
    c1 = candles[-1]

    if direction == "bullish":
        ok = is_bullish_order_block(c0, c1, require_displacement=require_displacement)

    elif direction == "bearish":
        ok = is_bearish_order_block(c0, c1, require_displacement=require_displacement)

    else:
        return Response({"ok": False, "reason": "invalid direction"})

    return Response({"ok": ok})