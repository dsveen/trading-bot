"""Walk-forward backtester for the swing trading strategy.

Replicates the same rules Claude is given in the system prompt:
  - RSI > 70 = overbought (no buy)
  - RSI < 30 = oversold (potential buy)
  - Volume above average required to enter
  - Stop-loss = 1.5 × ATR below entry
  - Take-profit = 1.5 × (stop distance) above entry  →  1.5:1 risk/reward

Entry is simulated at the NEXT bar's open price after a signal bar closes,
which is the realistic minimum latency for daily-bar strategies.
Stop and target are checked against that bar's low/high (intraday touch model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from trading_bot.indicators import compute_indicators


# ── Signal scoring ────────────────────────────────────────────────────────────

def _score_bar(snapshot) -> tuple[str, list[str]]:
    """Return (signal, reasons) where signal is BUY / SELL / HOLD.

    Mirrors the rules in analyst.py SYSTEM_PROMPT.
    """
    bull: list[str] = []
    bear: list[str] = []

    # Trend
    if snapshot.trend_label == "Bullish":
        bull.append("Uptrend (SMA20 > SMA50)")
    elif snapshot.trend_label == "Bearish":
        bear.append("Downtrend (SMA20 < SMA50)")

    # RSI
    if snapshot.rsi_14 > 70:
        bear.append(f"RSI overbought ({snapshot.rsi_14:.1f})")
    elif snapshot.rsi_14 < 30:
        bull.append(f"RSI oversold ({snapshot.rsi_14:.1f})")

    # MACD histogram direction
    if snapshot.macd_histogram > 0 and snapshot.macd_label == "Bullish":
        bull.append("MACD bullish crossover")
    elif snapshot.macd_histogram < 0 and snapshot.macd_label == "Bearish":
        bear.append("MACD bearish crossover")

    # Bollinger Bands
    if snapshot.bb_label == "Near Lower Band":
        bull.append("Price near BB lower band")
    elif snapshot.bb_label == "Near Upper Band":
        bear.append("Price near BB upper band")

    # Volume confirmation gate: require above-avg volume for any trade
    if snapshot.volume_label != "Above Average":
        return "HOLD", ["Volume below average — no trade"]

    # Need 3+ bullish or 3+ bearish signals
    if len(bull) >= 3 and len(bear) == 0:
        return "BUY", bull
    if len(bear) >= 3 and len(bull) == 0:
        return "SELL", bear
    return "HOLD", bull + bear


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    entry_date: str
    exit_date: str
    side: str           # "long" / "short"
    entry_price: float
    exit_price: float
    shares: int
    stop_loss: float
    take_profit: float
    exit_reason: str    # "stop_loss" / "take_profit" / "end_of_data"
    pl: float
    plpc: float
    equity_after: float
    signals: list[str] = field(default_factory=list)


# ── Core engine ───────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    capital: float = 10_000.0,
    position_pct: float = 0.10,
    stop_atr_mult: float = 1.5,
    rr_ratio: float = 1.5,
    allow_short: bool = False,
) -> dict:
    """Walk-forward simulation on a daily OHLCV DataFrame.

    Args:
        df:            Full OHLCV history (oldest bar first).
        symbol:        Ticker name, used for display only.
        capital:       Starting account equity.
        position_pct:  Fraction of equity to risk per trade (default 10%).
        stop_atr_mult: ATR multiplier for stop distance (default 1.5).
        rr_ratio:      Risk/reward target multiplier (default 1.5 → 2.25× ATR).
        allow_short:   Whether to also take SELL / short signals.

    Returns a dict with keys: trades, equity_curve, summary.
    """
    MIN_BARS = 60   # enough for SMA50 + warmup
    if len(df) < MIN_BARS + 2:
        return {"trades": [], "equity_curve": [capital], "summary": {}}

    trades: list[BacktestTrade] = []
    equity = capital
    equity_curve: list[float] = [capital]

    position: dict | None = None   # holds active trade state

    for i in range(MIN_BARS, len(df) - 1):
        window = df.iloc[: i + 1]
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        bar_date = str(bar.name)[:10]
        next_date = str(next_bar.name)[:10]

        try:
            snap = compute_indicators(window, symbol=symbol)
        except Exception:
            continue

        # ── CHECK EXIT on NEXT bar (entered at next_bar open) ─────────────
        if position is not None:
            lo = float(next_bar["low"])
            hi = float(next_bar["high"])

            hit_stop = (position["side"] == "long" and lo <= position["stop_loss"]) or \
                       (position["side"] == "short" and hi >= position["stop_loss"])
            hit_tp   = (position["side"] == "long" and hi >= position["take_profit"]) or \
                       (position["side"] == "short" and lo <= position["take_profit"])

            if hit_stop or hit_tp:
                reason = "take_profit" if hit_tp else "stop_loss"
                exit_px = position["take_profit"] if hit_tp else position["stop_loss"]

                if position["side"] == "long":
                    pl = (exit_px - position["entry"]) * position["shares"]
                else:
                    pl = (position["entry"] - exit_px) * position["shares"]

                plpc = pl / (position["entry"] * position["shares"])
                equity += pl
                equity_curve.append(round(equity, 2))

                trades.append(BacktestTrade(
                    symbol=symbol,
                    entry_date=position["entry_date"],
                    exit_date=next_date,
                    side=position["side"],
                    entry_price=position["entry"],
                    exit_price=exit_px,
                    shares=position["shares"],
                    stop_loss=position["stop_loss"],
                    take_profit=position["take_profit"],
                    exit_reason=reason,
                    pl=round(pl, 2),
                    plpc=round(plpc, 4),
                    equity_after=round(equity, 2),
                    signals=position["signals"],
                ))
                position = None

        # ── CHECK ENTRY signal on current bar, enter at NEXT bar open ─────
        if position is None:
            signal, reasons = _score_bar(snap)

            if signal == "BUY" or (allow_short and signal == "SELL"):
                side = "long" if signal == "BUY" else "short"
                entry_px = float(next_bar["open"])
                stop_dist = stop_atr_mult * snap.atr_14
                tp_dist   = stop_dist * rr_ratio

                if side == "long":
                    stop = entry_px - stop_dist
                    tp   = entry_px + tp_dist
                else:
                    stop = entry_px + stop_dist
                    tp   = entry_px - tp_dist

                pos_value = equity * position_pct
                shares = max(1, int(pos_value / entry_px))

                position = {
                    "side": side,
                    "entry": entry_px,
                    "entry_date": next_date,
                    "stop_loss": round(stop, 2),
                    "take_profit": round(tp, 2),
                    "shares": shares,
                    "signals": reasons,
                }

    # ── FORCE-CLOSE any open position at last bar close ───────────────────
    if position is not None:
        last_bar = df.iloc[-1]
        exit_px = float(last_bar["close"])
        last_date = str(last_bar.name)[:10]

        if position["side"] == "long":
            pl = (exit_px - position["entry"]) * position["shares"]
        else:
            pl = (position["entry"] - exit_px) * position["shares"]

        plpc = pl / (position["entry"] * position["shares"])
        equity += pl
        equity_curve.append(round(equity, 2))

        trades.append(BacktestTrade(
            symbol=symbol,
            entry_date=position["entry_date"],
            exit_date=last_date,
            side=position["side"],
            entry_price=position["entry"],
            exit_price=exit_px,
            shares=position["shares"],
            stop_loss=position["stop_loss"],
            take_profit=position["take_profit"],
            exit_reason="end_of_data",
            pl=round(pl, 2),
            plpc=round(plpc, 4),
            equity_after=round(equity, 2),
            signals=position["signals"],
        ))

    summary = _summarise(trades, capital, equity, equity_curve)
    return {"trades": trades, "equity_curve": equity_curve, "summary": summary}


# ── Statistics ────────────────────────────────────────────────────────────────

def _summarise(
    trades: list[BacktestTrade],
    initial: float,
    final: float,
    curve: list[float],
) -> dict:
    if not trades:
        return {}

    wins  = [t for t in trades if t.pl > 0]
    losses = [t for t in trades if t.pl <= 0]

    gross_profit = sum(t.pl for t in wins)
    gross_loss   = abs(sum(t.pl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win  = gross_profit / len(wins)  if wins   else 0.0
    avg_loss = gross_loss   / len(losses) if losses else 0.0

    # Max drawdown
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Simplified Sharpe (per-trade returns, assuming ~252 trading days/year)
    returns = [t.plpc for t in trades]
    if len(returns) > 1:
        import statistics
        mean_r = statistics.mean(returns)
        std_r  = statistics.stdev(returns)
        sharpe = (mean_r / std_r) * math.sqrt(len(returns)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       len(wins) / len(trades),
        "total_pl":       round(final - initial, 2),
        "total_return":   round((final - initial) / initial, 4),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 2),
        "max_drawdown":   round(max_dd, 4),
        "sharpe":         round(sharpe, 2),
        "final_equity":   round(final, 2),
    }
