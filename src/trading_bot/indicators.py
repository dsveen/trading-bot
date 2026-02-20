import pandas as pd
import ta

from trading_bot.models import IndicatorSnapshot


def compute_indicators(
    df: pd.DataFrame, symbol: str, fast: bool = False
) -> IndicatorSnapshot:
    """Compute technical indicators.

    Args:
        fast: Use short-period windows for scalping (1-min bars).
              Default False uses standard windows (daily bars).
    """
    close = df["close"]
    volume = df["volume"]

    # Window sizes: fast (scalp on 1-min bars) vs standard (daily)
    if fast:
        sma_short_w, sma_long_w = 5, 13
        ema_fast_w, ema_slow_w = 5, 13
        rsi_w = 7
        macd_fast_w, macd_slow_w, macd_sig_w = 5, 13, 4
        bb_w = 10
        atr_w = 7
        vol_avg_w = 10
    else:
        sma_short_w, sma_long_w = 20, 50
        ema_fast_w, ema_slow_w = 12, 26
        rsi_w = 14
        macd_fast_w, macd_slow_w, macd_sig_w = 12, 26, 9
        bb_w = 20
        atr_w = 14
        vol_avg_w = 20

    # SMAs and EMAs
    sma_short = ta.trend.sma_indicator(close, window=sma_short_w).iloc[-1]
    sma_long = ta.trend.sma_indicator(close, window=sma_long_w).iloc[-1]
    ema_fast = ta.trend.ema_indicator(close, window=ema_fast_w).iloc[-1]
    ema_slow = ta.trend.ema_indicator(close, window=ema_slow_w).iloc[-1]

    # RSI
    rsi = ta.momentum.rsi(close, window=rsi_w).iloc[-1]

    # MACD
    macd = ta.trend.MACD(
        close, window_fast=macd_fast_w, window_slow=macd_slow_w, window_sign=macd_sig_w
    )
    macd_line = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]
    macd_histogram = macd.macd_diff().iloc[-1]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=bb_w, window_dev=2)
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_middle = bb.bollinger_mavg().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]

    # ATR
    atr = ta.volatility.average_true_range(
        df["high"], df["low"], close, window=atr_w
    ).iloc[-1]

    # Volume
    current_price = float(close.iloc[-1])
    volume_current = int(volume.iloc[-1])
    volume_avg = float(volume.rolling(window=vol_avg_w).mean().iloc[-1])

    # Interpretive labels
    if rsi >= 70:
        rsi_label = "OVERBOUGHT"
    elif rsi <= 30:
        rsi_label = "OVERSOLD"
    else:
        rsi_label = "NEUTRAL"

    macd_label = f"{'Bullish' if ema_fast > ema_slow else 'Bearish'}"

    price_pct = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    if price_pct > 0.8:
        bb_label = "Near Upper Band"
    elif price_pct < 0.2:
        bb_label = "Near Lower Band"
    else:
        bb_label = "Middle Range"

    if sma_short > sma_long and current_price > sma_short:
        trend_label = "Bullish"
    elif sma_short < sma_long and current_price < sma_short:
        trend_label = "Bearish"
    else:
        trend_label = "Neutral"

    volume_label = "Above Average" if volume_current > volume_avg else "Below Average"

    return IndicatorSnapshot(
        symbol=symbol,
        current_price=round(current_price, 2),
        sma_20=round(float(sma_short), 2),
        sma_50=round(float(sma_long), 2),
        ema_12=round(float(ema_fast), 2),
        ema_26=round(float(ema_slow), 2),
        rsi_14=round(float(rsi), 2),
        macd_line=round(float(macd_line), 4),
        macd_signal=round(float(macd_signal), 4),
        macd_histogram=round(float(macd_histogram), 4),
        bb_upper=round(float(bb_upper), 2),
        bb_middle=round(float(bb_middle), 2),
        bb_lower=round(float(bb_lower), 2),
        atr_14=round(float(atr), 2),
        volume_current=volume_current,
        volume_avg_20=round(volume_avg, 0),
        rsi_label=rsi_label,
        macd_label=macd_label,
        bb_label=bb_label,
        trend_label=trend_label,
        volume_label=volume_label,
    )
