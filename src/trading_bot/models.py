from enum import Enum
from pydantic import BaseModel, Field


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeDecision(BaseModel):
    action: Action
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    suggested_shares: int = Field(ge=0)
    suggested_entry_price: float = Field(ge=0)
    stop_loss: float = Field(ge=0)
    take_profit: float = Field(ge=0)
    risk_reward_ratio: float = Field(ge=0)
    time_horizon: str


class IndicatorSnapshot(BaseModel):
    symbol: str
    current_price: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr_14: float
    volume_current: int
    volume_avg_20: float

    # Interpretive labels
    rsi_label: str  # OVERSOLD / NEUTRAL / OVERBOUGHT
    macd_label: str  # Bullish / Bearish
    bb_label: str  # Near Upper / Middle / Near Lower
    trend_label: str  # Bullish / Bearish / Neutral
    volume_label: str  # Above Average / Below Average


class SafetyCheckResult(BaseModel):
    passed: bool
    reason: str | None = None
