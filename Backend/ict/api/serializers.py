from rest_framework import serializers
from ict.api.models import *
from datetime import timezone
from decimal import Decimal, InvalidOperation

utc = timezone.utc

# Serializer for CustomUser model
class CustomUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = "__all__"

# Serializer for Role model
class RoleSerializers(serializers.ModelSerializer):
    class Meta:
        model = Role
        fields = "__all__"

# Serializer for Instrument model
class InstrumentSerializers(serializers.ModelSerializer):
    class Meta:
        model = Instrument
        fields = "__all__"

# Serializer for Instrument model
class CandleSerializers(serializers.ModelSerializer):
    instrument = InstrumentSerializers(read_only=True)
    
    class Meta:
        model = Candle
        fields = "__all__"

# Serializer for Flashcard model
class FlashCardSerializers(serializers.ModelSerializer):
    
    class Meta:
        model = Flashcard
        fields = "__all__" 


class TradeJournalSerializer(serializers.ModelSerializer):
    username = serializers.CharField()  # accept string username
    class Meta:
        model = TradeJournal
        fields = "__all__"

    def validate_username(self, value):
        from django.contrib.auth import get_user_model
        User = get_user_model()

        try:
            user = User.objects.get(username=value)
        except User.DoesNotExist:
            raise serializers.ValidationError("User not found.")

        return user   # return the actual user object
    def create(self, validated_data):
        # username is a User object because validate_username returned it
        return super().create(validated_data)
        
        extra_kwargs = {
            "risk_percent": {"required": False, "allow_null": True},
            "r_multiple": {"required": False, "allow_null": True},
            "account_balance": {"required": False, "allow_null": True},
            "risk_amount": {"required": False, "allow_null": True},
        }

    def to_internal_value(self, data):
        """
        Clean up empty strings coming from the frontend:
        '' -> None for numeric fields.
        """
        data = data.copy()
        for field in ["risk_percent", "r_multiple", "account_balance", "risk_amount"]:
            if field in data and data[field] == "":
                data[field] = None
        return super().to_internal_value(data)

    def _compute_risk_fields(self, validated_data):
        """
        Optional auto-calc:
          risk_percent = (risk_amount / account_balance) * 100
          r_multiple   = pnl / risk_amount
        Only fills them if they're missing/zero and enough info exists.
        """
        account_balance = validated_data.get("account_balance")
        risk_amount = validated_data.get("risk_amount")
        pnl = validated_data.get("pnl")

        risk_percent = validated_data.get("risk_percent")
        r_multiple = validated_data.get("r_multiple")

        # ---- Risk % ----
        try:
            if (
                account_balance is not None
                and risk_amount is not None
                and Decimal(str(account_balance)) != 0
            ):
                if risk_percent in (None, Decimal("0"), Decimal("0.00")):
                    rp = (
                        Decimal(str(risk_amount))
                        / Decimal(str(account_balance))
                        * Decimal("100")
                    )
                    validated_data["risk_percent"] = rp.quantize(Decimal("0.01"))
        except (InvalidOperation, ZeroDivisionError):
            pass

        # ---- R-multiple ----
        try:
            if (
                risk_amount not in (None, Decimal("0"), Decimal("0.00"))
                and pnl is not None
            ):
                if r_multiple in (None, Decimal("0"), Decimal("0.00")):
                    rm = Decimal(str(pnl)) / Decimal(str(risk_amount))
                    validated_data["r_multiple"] = rm.quantize(Decimal("0.01"))
        except (InvalidOperation, ZeroDivisionError):
            pass

        return validated_data

    def create(self, validated_data):
        validated_data = self._compute_risk_fields(validated_data)
        return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data = self._compute_risk_fields(validated_data)
        return super().update(instance, validated_data)
