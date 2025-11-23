from django.urls import path, include
from .views import *

urlpatterns = [
    path('test/', test, name='test'),
    path('get_data/', get_data, name='get_data'),
    path('fvg_data/', fvg_data, name='fvg_data'),
    path('check_fvg/', check_fvg, name='check_fvg'),
    path('check_order_block/', check_order_block, name='check_order_block'),
]
    