import random
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from ...models import Candle, Instrument


TIMEFRAMES = [
    ("1m", 1),
    ("3m", 3),
    ("5m", 5),
    ("15m", 15),
]


class Command(BaseCommand):
    help = "Generate synthetic ICT-style candles with realistic volatility"

    def add_arguments(self, parser):
        parser.add_argument(
            "--candles_per_symbol",
            type=int,
            default=500,
            help="How many candles per symbol"
        )

    def handle(self, *args, **options):
        candles_per_symbol = options["candles_per_symbol"]

        self.stdout.write(self.style.NOTICE(
            f"Generating {candles_per_symbol} candles PER SYMBOL with mixed timeframes..."
        ))

        symbols = [
            "AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
            "META", "NFLX", "NVDA", "AMD", "INTC",
            "SPY", "QQQ", "IWM", "NQ100", "ES500",
            "US30", "GC", "CL", "EURUSD", "GBPUSD"
        ]

        for sym in symbols:
            inst, _ = Instrument.objects.get_or_create(symbol=sym, defaults={"name": sym})
            self._generate_for_symbol(inst, candles_per_symbol)

        self.stdout.write(self.style.SUCCESS("Done generating synthetic candles."))

    # ------------------------------------------------------------------
    #              REALISTIC ICT-STYLE CANDLE GENERATOR
    # ------------------------------------------------------------------
    def _generate_for_symbol(self, instrument, n):
        timeframe, minutes_per_candle = random.choice(TIMEFRAMES)

        end_time = timezone.now()
        start_time = end_time - timedelta(minutes=minutes_per_candle * n)

        # Base price per symbol – realistic + unique
        base_price = random.uniform(60, 180)
        price = base_price

        candles = []
        current_time = start_time

        # Volatility (percent of price)
        small_vol_min = 0.0005  # 0.05%
        small_vol_max = 0.0025  # 0.25%

        big_vol_min = 0.004     # 0.4%
        big_vol_max = 0.012     # 1.2%

        for i in range(n):
            # Default small volatility
            vol_pct = random.uniform(small_vol_min, small_vol_max)

            # Every ~40 candles → session push
            if i % 40 in (0, 1, 2, 3):
                vol_pct = random.uniform(big_vol_min, big_vol_max)

            # ICT displacement candle
            is_displacement = (i % 60 == 0)

            direction = 1 if random.random() < 0.5 else -1

            o = price

            if is_displacement:
                move_pct = vol_pct * random.uniform(3.0, 5.0)
            else:
                move_pct = vol_pct * random.uniform(0.5, 1.5)

            c = o * (1 + direction * move_pct)

            # Wicks
            wick_pct = vol_pct * random.uniform(0.3, 1.0)
            h = max(o, c) * (1 + wick_pct)
            l = min(o, c) * (1 - wick_pct)

            # Liquidity grab wicks every 25 candles
            if i % 25 == 0:
                liq_pct = vol_pct * random.uniform(2.0, 4.0)
                if random.random() < 0.5:
                    h = max(h, max(o, c) * (1 + liq_pct))
                else:
                    l = min(l, min(o, c) * (1 - liq_pct))

            # Volume
            vol = random.randint(5_000, 50_000)
            if i % 35 in (0, 1):
                vol *= random.randint(2, 4)

            candles.append(
                Candle(
                    instrument=instrument,
                    timestamp=current_time,
                    timeframe=timeframe,
                    _open=round(o, 2),
                    high=round(h, 2),
                    low=round(l, 2),
                    close=round(c, 2),
                    volume=vol,
                )
            )

            price = c
            current_time += timedelta(minutes=minutes_per_candle)

        Candle.objects.bulk_create(candles, batch_size=1000)