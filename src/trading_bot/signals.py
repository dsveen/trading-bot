"""Code-based signal scanner.

Runs BEFORE Claude to avoid wasting API calls on no-signal ticks.
Also handles mechanical exits (trailing stop, profit target, time limit)
so we don't rely on the AI for fast exit decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime

from trading_bot.models import IndicatorSnapshot


@dataclass
class Signal:
    direction: str  # "long", "short", or "none"
    strength: int  # 0-5, number of confirming signals
    reasons: list[str] = field(default_factory=list)


def scan_entry(snapshot: IndicatorSnapshot) -> Signal:
    """Check for high-probability entry setups.

    Only returns a signal when multiple indicators align.
    Requires at least 3 confirming factors to suggest an entry.
    """
    long_points = 0
    short_points = 0
    long_reasons = []
    short_reasons = []

    # 1. RSI extremes (mean reversion)
    if snapshot.rsi_14 < 30:
        long_points += 2
        long_reasons.append(f"RSI oversold ({snapshot.rsi_14:.1f})")
    elif snapshot.rsi_14 < 40:
        long_points += 1
        long_reasons.append(f"RSI low ({snapshot.rsi_14:.1f})")
    elif snapshot.rsi_14 > 70:
        short_points += 2
        short_reasons.append(f"RSI overbought ({snapshot.rsi_14:.1f})")
    elif snapshot.rsi_14 > 60:
        short_points += 1
        short_reasons.append(f"RSI high ({snapshot.rsi_14:.1f})")

    # 2. MACD histogram direction (momentum)
    if snapshot.macd_histogram > 0 and snapshot.macd_line > snapshot.macd_signal:
        long_points += 1
        long_reasons.append("MACD bullish crossover")
    elif snapshot.macd_histogram < 0 and snapshot.macd_line < snapshot.macd_signal:
        short_points += 1
        short_reasons.append("MACD bearish crossover")

    # 3. Bollinger Band position (mean reversion)
    if snapshot.bb_label == "Near Lower Band":
        long_points += 1
        long_reasons.append("Price near lower BB")
    elif snapshot.bb_label == "Near Upper Band":
        short_points += 1
        short_reasons.append("Price near upper BB")

    # 4. Trend alignment (SMA direction)
    if snapshot.trend_label == "Bullish":
        long_points += 1
        long_reasons.append("SMA trend bullish")
    elif snapshot.trend_label == "Bearish":
        short_points += 1
        short_reasons.append("SMA trend bearish")

    # 5. Volume confirmation (REQUIRED for any entry)
    has_volume = snapshot.volume_label == "Above Average"
    if not has_volume:
        # No volume = no trade, regardless of other signals
        return Signal(direction="none", strength=0, reasons=["No volume confirmation"])

    # Need at least 3 confirming signals to enter
    if long_points >= 3:
        long_reasons.append("Volume confirmed")
        return Signal(direction="long", strength=long_points, reasons=long_reasons)
    elif short_points >= 3:
        short_reasons.append("Volume confirmed")
        return Signal(direction="short", strength=short_points, reasons=short_reasons)

    return Signal(direction="none", strength=max(long_points, short_points),
                  reasons=["Insufficient confluence"])


@dataclass
class PositionTracker:
    """Mechanical exit management — no AI needed for exits."""

    entry_price: float = 0.0
    side: str = ""  # "long" or "short"
    entry_time: datetime | None = None
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")
    atr_at_entry: float = 0.0

    # Exit parameters
    profit_target_atr: float = 2.0   # take profit at 2x ATR
    stop_loss_atr: float = 1.0       # stop loss at 1x ATR
    trailing_stop_atr: float = 1.5   # trail at 1.5x ATR from peak
    max_hold_minutes: int = 10       # force exit after 10 minutes

    def enter(self, price: float, side: str, atr: float) -> None:
        self.entry_price = price
        self.side = side
        self.entry_time = datetime.now()
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.atr_at_entry = atr

    def clear(self) -> None:
        self.entry_price = 0.0
        self.side = ""
        self.entry_time = None
        self.highest_since_entry = 0.0
        self.lowest_since_entry = float("inf")

    @property
    def is_active(self) -> bool:
        return self.side != ""

    def check_exit(self, current_price: float) -> tuple[bool, str]:
        """Check if we should exit mechanically. Returns (should_exit, reason)."""
        if not self.is_active:
            return False, ""

        # Update high/low watermarks
        self.highest_since_entry = max(self.highest_since_entry, current_price)
        self.lowest_since_entry = min(self.lowest_since_entry, current_price)

        atr = self.atr_at_entry

        if self.side == "long":
            # Profit target
            target = self.entry_price + (self.profit_target_atr * atr)
            if current_price >= target:
                return True, f"Profit target hit (${target:.2f})"

            # Hard stop loss
            stop = self.entry_price - (self.stop_loss_atr * atr)
            if current_price <= stop:
                return True, f"Stop loss hit (${stop:.2f})"

            # Trailing stop from highest point
            trail_stop = self.highest_since_entry - (self.trailing_stop_atr * atr)
            if current_price <= trail_stop and self.highest_since_entry > self.entry_price:
                return True, f"Trailing stop hit (${trail_stop:.2f} from peak ${self.highest_since_entry:.2f})"

        elif self.side == "short":
            # Profit target
            target = self.entry_price - (self.profit_target_atr * atr)
            if current_price <= target:
                return True, f"Profit target hit (${target:.2f})"

            # Hard stop loss
            stop = self.entry_price + (self.stop_loss_atr * atr)
            if current_price >= stop:
                return True, f"Stop loss hit (${stop:.2f})"

            # Trailing stop from lowest point
            trail_stop = self.lowest_since_entry + (self.trailing_stop_atr * atr)
            if current_price >= trail_stop and self.lowest_since_entry < self.entry_price:
                return True, f"Trailing stop hit (${trail_stop:.2f} from low ${self.lowest_since_entry:.2f})"

        # Time-based exit
        if self.entry_time:
            held_seconds = (datetime.now() - self.entry_time).total_seconds()
            if held_seconds > self.max_hold_minutes * 60:
                return True, f"Time limit ({self.max_hold_minutes}min)"

        return False, ""
