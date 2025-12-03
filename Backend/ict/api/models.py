from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token

class Role(models.Model):
    """
    Model representing different user roles within the system.
    
    Attributes:
        name (str): The name of the role, ensuring uniqueness across records.
    """
    name = models.CharField(max_length=30, unique=True)

    def __str__(self):
        return self.name 

class CustomUser(AbstractUser):
    """
    Extended user model that includes additional attributes for user customization and roles.
    
    Attributes:
        color (str): A hex color code representing the user profile.
        profile_access (ManyToManyField): Assignable roles for user permissions.
        phone_number (str): Optional phone number for the user.
        wilderness_track (ManyToManyField): Tracks associated with the user.
        gender (str): Gender of the user.
     
    Methods: 
        save: Overrides the default save method to automatically assign a default role ('Pending Approval') upon user creation.
    """
    color = models.TextField(null=False, blank=False, default='#000')
    profile_access = models.ManyToManyField(Role, blank=True, related_name='users')
    phone_number = models.TextField(null=True, blank=True, unique=True)
    gender = models.TextField(null=True, blank=True)
    email = models.EmailField(unique=True)
    # newFeatureViewed = models.BooleanField(blank=False, default=False)
    

class Instrument(models.Model):
    symbol = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.symbol


class Candle(models.Model):
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, related_name="candles")
    timestamp = models.DateTimeField()
    timeframe = models.CharField(max_length=10, default="5m")  # 1m, 5m, 15m, etc.
    _open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()

    class Meta:
        indexes = [
            models.Index(fields=["instrument", "timestamp"]),
        ]

    def __str__(self):
        return f"{self.instrument.symbol} {self.timestamp} {self.timeframe}"
        

class TradeJournal(models.Model):
    DIRECTION_CHOICES = [
        ("long", "Long"),
        ("short", "Short"),
    ]

    SESSION_CHOICES = [
        ("london", "London"),
        ("newyork", "New York"),
        ("asia", "Asia"),
    ]

    RESULT_CHOICES = [
        ("win", "Win"),
        ("loss", "Loss"),
        ("breakeven", "Break Even"),
    ]

    TIMEFRAME_CHOICES = [
        ("1m", "1m"),
        ("3m", "3m"),
        ("5m", "5m"),
        ("15m", "15m"),
        ("1h", "1H"),
        ("4h", "4H"),
        ("d", "D"),
    ]

    GRADE_CHOICES = [
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("D", "D"),
    ]

    # core trade info
    date = models.DateField()
    time = models.TimeField()
    symbol = models.CharField(max_length=20)
    direction = models.CharField(max_length=5, choices=DIRECTION_CHOICES)
    session = models.CharField(max_length=20, choices=SESSION_CHOICES)
    timeframe = models.CharField(max_length=10, choices=TIMEFRAME_CHOICES)

    # risk fields (we'll auto-compute risk_percent if possible)
    account_balance = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )
    risk_amount = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True
    )
    risk_percent = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
    )

    r_multiple = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
    )

    entry_price = models.DecimalField(max_digits=12, decimal_places=2)
    exit_price = models.DecimalField(max_digits=12, decimal_places=2)

    result = models.CharField(max_length=10, choices=RESULT_CHOICES)
    pnl = models.DecimalField(max_digits=10, decimal_places=2)

    ict_setup = models.CharField(max_length=255, blank=True)
    higher_tf_bias = models.CharField(max_length=20, blank=True)

    # ICT flags
    ifvg_used = models.BooleanField(default=False)
    ob_used = models.BooleanField(default=False)
    breaker_used = models.BooleanField(default=False)
    liquidity_sweep_used = models.BooleanField(default=False)
    mitigation_block_used = models.BooleanField(default=False)

    # psychology
    pre_trade_emotion = models.CharField(max_length=50, blank=True)
    post_trade_emotion = models.CharField(max_length=50, blank=True)
    trade_grade = models.CharField(max_length=2, choices=GRADE_CHOICES, blank=True)

    followed_plan = models.BooleanField(default=False)
    emotional_trade = models.BooleanField(default=False)
    took_profit_early = models.BooleanField(default=False)
    missed_trade = models.BooleanField(default=False)
    moved_stop_loss = models.BooleanField(default=False)
    revenge_trade = models.BooleanField(default=False)

    notes = models.TextField(blank=True)

    # complex levels as JSON
    stop_levels = models.JSONField(default=list, blank=True)
    take_profit_levels = models.JSONField(default=list, blank=True)

    # optional external trade id (e.g. TopstepX)
    external_trade_id = models.CharField(
        max_length=64, blank=True, null=True, unique=True
    )

    created_at = models.DateTimeField(auto_now_add=True)
    username = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    def __str__(self):
        return f"Trade #{self.pk} {self.symbol} {self.direction} {self.date}"

class Flashcard(models.Model):
    question = models.TextField()
    answer = models.TextField()
    reasoning = models.TextField()
    course = models.TextField()

    def __str__(self):
        return f"{self.course} - {self.question}"

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)