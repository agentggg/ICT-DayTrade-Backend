from django.urls import path, include
from .views import *

urlpatterns = [
    path('test/', test, name='test'),
    path('get_data/', get_data, name='get_data'),
    path('get_candles/', get_candles, name='get_candles'),
    path('check_fvg/', check_fvg, name='check_fvg'),
    path('check_order_block/', check_order_block, name='check_order_block'),
    path('check_breaker_block/', check_breaker_block, name='check_breaker_block'),
    path('login_verification/', login_verification, name='login_verification'),
    path("trades", trade_journal_view, name="trade-journal"),
    path("flashcard", flashcard, name="flashcard"),
    path("get_flashcard", get_flashcard, name="get_flashcard"),
    path("predict/", predict, name="predict"),
    path("recognize/", recognize, name="recognize"),
    path("enroll/", enroll, name="enroll"),
    path("hand_landmarks/", hand_recognition, name="hand_landmarks"),
    path("gesture_recognition/", gesture_recognition, name="gesture_recognition"),
    path("object_gesture/", object_gesture, name="object_gesture"),
    # path("recognition/", recognition, name="recognition"),
    path("hand_recognition/", hand_recognition, name="hand_recognition"),
    

]
      