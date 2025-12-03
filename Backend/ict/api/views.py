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

<<<<<<< HEAD
    # Get all 
    # user = asdsd
    
    trades = TradeJournal.objects.all().order_by("-date", "-time")
=======
    # Get all
    user = request.GET.get("username")

    if not user:
        return Response({"detail": "Username is required."}, status=400)

    try:
        userInstance = CustomUser.objects.get(username=user)
    except CustomUser.DoesNotExist:
        return Response({"detail": "User not found."}, status=404)

    trades = TradeJournal.objects.filter(username=userInstance).order_by("-date", "-time")
>>>>>>> 760382ccde8258f3a90eda99881cbaecf49ca73b
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
    """
    """
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
    """

    """
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
