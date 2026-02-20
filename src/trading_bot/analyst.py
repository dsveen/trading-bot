import anthropic

from trading_bot.config import Config
from trading_bot.models import IndicatorSnapshot, TradeDecision

# Standard swing analysis prompt
SYSTEM_PROMPT = """\
Stock trading analyst. Return a structured trade decision from technical indicators.
Rules: RSI>70=overbought, RSI<30=oversold. Require volume above avg for BUY/SELL. \
Min risk/reward 1.5:1. Stop-loss=1.5*ATR from entry. When signals conflict, HOLD. \
Max position $1000. Entry near current price. \
time_horizon: "intraday" | "swing (2-5 days)" | "position (1-4 weeks)"."""

# Scalping prompt — called ONLY when code-level scanner already found a setup
SCALP_SYSTEM_PROMPT = """\
Scalp trader confirming a pre-screened setup on 1-min bars. \
Code detected {signal_direction} signal (strength {signal_strength}/5): {signal_reasons}. \
Your job: confirm or reject this entry. Only say BUY/SELL if you agree the setup is strong. \
Say HOLD if anything looks wrong (divergence, fading momentum, overhead resistance, thin volume). \
Stop-loss=1*ATR from entry. Take-profit=2*ATR from entry. Risk/reward must be >=1.5. \
Max position $1000. Entry=current price. time_horizon="scalp (1-5 min)". \
Be selective. A rejected bad trade is better than a losing trade."""


def build_prompt(snapshot: IndicatorSnapshot) -> str:
    """Build a minimal token-efficient prompt from indicators."""
    return (
        f"{snapshot.symbol} ${snapshot.current_price}\n"
        f"SMA5={snapshot.sma_20} SMA13={snapshot.sma_50} "
        f"EMA5={snapshot.ema_12} EMA13={snapshot.ema_26} [{snapshot.trend_label}]\n"
        f"RSI7={snapshot.rsi_14} [{snapshot.rsi_label}]\n"
        f"MACD={snapshot.macd_line}/{snapshot.macd_signal}/{snapshot.macd_histogram} [{snapshot.macd_label}]\n"
        f"BB={snapshot.bb_lower}/{snapshot.bb_middle}/{snapshot.bb_upper} [{snapshot.bb_label}]\n"
        f"ATR={snapshot.atr_14}\n"
        f"Vol={snapshot.volume_current:,} avg={snapshot.volume_avg_20:,.0f} [{snapshot.volume_label}]"
    )


def get_analysis(config: Config, snapshot: IndicatorSnapshot) -> TradeDecision:
    """Standard swing-trade analysis."""
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    response = client.messages.parse(
        model=config.claude_model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(snapshot)}],
        output_format=TradeDecision,
    )

    return response.parsed_output


def get_scalp_analysis(
    config: Config,
    snapshot: IndicatorSnapshot,
    signal_direction: str,
    signal_strength: int,
    signal_reasons: list[str],
) -> TradeDecision:
    """Scalp analysis — only called when code scanner finds a setup.

    Claude acts as a second opinion to confirm or reject.
    """
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    system = SCALP_SYSTEM_PROMPT.format(
        signal_direction=signal_direction.upper(),
        signal_strength=signal_strength,
        signal_reasons=", ".join(signal_reasons),
    )

    response = client.messages.parse(
        model=config.claude_model,
        max_tokens=384,
        system=system,
        messages=[{"role": "user", "content": build_prompt(snapshot)}],
        output_format=TradeDecision,
    )

    return response.parsed_output
