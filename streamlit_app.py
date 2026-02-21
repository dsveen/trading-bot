"""Streamlit web UI for the AI Trading Bot.

Tabs:
  📊 Backtest   — Walk-forward simulation on historical daily bars
  🔍 Analyze    — Fetch live indicators + Claude AI recommendation
  💼 Portfolio  — Open positions and trade history
  🤖 AI Review  — Claude verdict on every open position
"""

import sys
import os

# Allow imports from src/ layout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from trading_bot.config import Config
from trading_bot.market_data import (
    fetch_bars,
    fetch_account,
    fetch_positions,
    fetch_trade_history,
)
from trading_bot.indicators import compute_indicators
from trading_bot.analyst import get_analysis
from trading_bot.backtester import run_backtest

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar — credentials & settings ──────────────────────────────────────────

st.sidebar.title("⚙️ Settings")

# Load from Streamlit Cloud secrets if available, else show text inputs
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return default

with st.sidebar.expander("🔑 API Keys", expanded=not bool(_secret("ALPACA_API_KEY"))):
    alpaca_key = st.text_input(
        "Alpaca API Key",
        value=_secret("ALPACA_API_KEY"),
        type="password",
        help="From alpaca.markets → API Keys",
    )
    alpaca_secret = st.text_input(
        "Alpaca Secret Key",
        value=_secret("ALPACA_SECRET_KEY"),
        type="password",
    )
    anthropic_key = st.text_input(
        "Anthropic API Key",
        value=_secret("ANTHROPIC_API_KEY"),
        type="password",
        help="From console.anthropic.com",
    )

trading_mode = st.sidebar.radio(
    "Trading Mode",
    ["Paper", "Live"],
    index=0,
    help="Paper mode uses simulated money — safe for testing.",
)

claude_model = st.sidebar.selectbox(
    "Claude Model",
    ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"],
    index=0,
    help="Haiku is fastest and cheapest; Opus is most analytical.",
)

st.sidebar.divider()
st.sidebar.caption("Prices are delayed on free Alpaca tier (IEX, ~15 min).")


def make_config() -> Config | None:
    """Build a Config from sidebar values. Returns None if keys are missing."""
    if not alpaca_key or not alpaca_secret or not anthropic_key:
        return None
    return Config(
        alpaca_api_key=alpaca_key,
        alpaca_secret_key=alpaca_secret,
        anthropic_api_key=anthropic_key,
        trading_mode=trading_mode.lower(),
        auto_execute=False,
        claude_model=claude_model,
        max_position_size=1000.0,
        max_daily_loss=500.0,
        min_confidence=0.30,
    )


def require_config() -> Config:
    cfg = make_config()
    if cfg is None:
        st.warning("Enter your API keys in the sidebar to continue.")
        st.stop()
    return cfg


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_backtest, tab_analyze, tab_portfolio, tab_review = st.tabs(
    ["📊 Backtest", "🔍 Analyze", "💼 Portfolio", "🤖 AI Review"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.header("📊 Walk-Forward Backtest")
    st.caption(
        "Simulates the bot's rule-based swing strategy on historical daily bars. "
        "Entry at next bar's open; stop/target checked against intraday high/low."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        bt_symbol = st.text_input("Symbol", value="AAPL", key="bt_symbol").upper().strip()

    with col2:
        bt_days = st.number_input("History (days)", min_value=90, max_value=1825, value=365, step=30)

    with st.expander("⚙️ Strategy Parameters"):
        pc1, pc2, pc3, pc4 = st.columns(4)
        bt_capital = pc1.number_input("Starting Capital ($)", min_value=1000, value=10000, step=1000)
        bt_pos_pct = pc2.slider("Position Size (%)", min_value=5, max_value=50, value=10) / 100
        bt_stop = pc3.slider("Stop ATR Multiplier", min_value=0.5, max_value=4.0, value=1.5, step=0.1)
        bt_rr = pc4.slider("Risk/Reward Ratio", min_value=1.0, max_value=4.0, value=1.5, step=0.1)
        bt_short = st.checkbox("Allow Short Trades (SELL signals)", value=False)

    run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if run_bt:
        cfg = require_config()
        with st.spinner(f"Fetching {bt_days} days of {bt_symbol} bars…"):
            try:
                df = fetch_bars(cfg, bt_symbol, days=bt_days)
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                st.stop()

        with st.spinner("Running simulation…"):
            result = run_backtest(
                df=df,
                symbol=bt_symbol,
                capital=bt_capital,
                position_pct=bt_pos_pct,
                stop_atr_mult=bt_stop,
                rr_ratio=bt_rr,
                allow_short=bt_short,
            )

        summary = result["summary"]
        trades = result["trades"]
        curve = result["equity_curve"]

        if not summary:
            st.info("No trades were generated. Try a longer history or different parameters.")
        else:
            # ── Summary metrics ──────────────────────────────────────────────
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            total_return_pct = summary["total_return"] * 100
            m1.metric("Total Return", f"{total_return_pct:+.1f}%")
            m2.metric("Win Rate", f"{summary['win_rate']*100:.0f}%")
            m3.metric("Total Trades", summary["total_trades"])
            m4.metric("Profit Factor", f"{summary['profit_factor']:.2f}")
            m5.metric("Max Drawdown", f"{summary['max_drawdown']*100:.1f}%")
            m6.metric("Sharpe (simple)", f"{summary['sharpe']:.2f}")

            st.divider()

            # ── Equity curve ─────────────────────────────────────────────────
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=curve,
                    mode="lines",
                    name="Equity",
                    line=dict(color="#00D4AA", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,212,170,0.08)",
                )
            )
            # Baseline
            fig.add_hline(y=bt_capital, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_layout(
                title=f"{bt_symbol} Equity Curve",
                xaxis_title="Trade #",
                yaxis_title="Portfolio Value ($)",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── P&L distribution ─────────────────────────────────────────────
            pls = [t.pl for t in trades]
            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    y=pls,
                    marker_color=["#00D4AA" if p >= 0 else "#FF4B4B" for p in pls],
                    name="Trade P&L",
                )
            )
            fig2.update_layout(
                title="Individual Trade P&L ($)",
                xaxis_title="Trade #",
                yaxis_title="P&L ($)",
                height=250,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA"),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── Trade table ──────────────────────────────────────────────────
            with st.expander(f"📋 All {len(trades)} Trades"):
                rows = []
                for t in trades:
                    rows.append({
                        "Entry": t.entry_date,
                        "Exit": t.exit_date,
                        "Side": t.side,
                        "Entry $": f"{t.entry_price:.2f}",
                        "Exit $": f"{t.exit_price:.2f}",
                        "Shares": t.shares,
                        "P&L $": f"{t.pl:+.2f}",
                        "P&L %": f"{t.plpc*100:+.1f}%",
                        "Reason": t.exit_reason,
                        "Equity After": f"${t.equity_after:,.0f}",
                    })
                trades_df = pd.DataFrame(rows)

                def color_pl(val):
                    if isinstance(val, str) and val.startswith(("+", "-")):
                        color = "#00D4AA" if val.startswith("+") else "#FF4B4B"
                        return f"color: {color}"
                    return ""

                st.dataframe(
                    trades_df.style.applymap(color_pl, subset=["P&L $", "P&L %"]),
                    use_container_width=True,
                    hide_index=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyze:
    st.header("🔍 Live Analysis")
    st.caption("Fetches current daily bars, computes technical indicators, and asks Claude for a recommendation.")

    az_col1, az_col2 = st.columns([3, 1])
    az_symbol = az_col1.text_input("Symbol", value="AAPL", key="az_symbol").upper().strip()
    az_days = az_col2.number_input("Bar history (days)", min_value=60, max_value=365, value=100)

    run_az = st.button("🔍 Analyze", type="primary", use_container_width=True, key="btn_analyze")

    if run_az:
        cfg = require_config()
        with st.spinner(f"Fetching bars for {az_symbol}…"):
            try:
                df = fetch_bars(cfg, az_symbol, days=az_days)
            except Exception as e:
                st.error(f"Data fetch failed: {e}")
                st.stop()

        with st.spinner("Computing indicators…"):
            snap = compute_indicators(df, symbol=az_symbol)

        with st.spinner("Asking Claude…"):
            decision = get_analysis(cfg, snap)

        # ── Action banner ────────────────────────────────────────────────────
        action_colors = {"BUY": "#00D4AA", "SELL": "#FF4B4B", "HOLD": "#FFA500"}
        action_color = action_colors.get(decision.action, "#888888")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {action_color}22, {action_color}11);
                border: 2px solid {action_color};
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
            ">
                <h2 style="color:{action_color}; margin:0; font-size:2rem;">
                    {decision.action}
                </h2>
                <p style="margin:8px 0 0 0; color:#FAFAFA; font-size:1rem;">
                    Confidence: {decision.confidence_score:.0%} &nbsp;|&nbsp;
                    Time horizon: {decision.time_horizon} &nbsp;|&nbsp;
                    Risk/Reward: {decision.risk_reward_ratio:.1f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Claude's reasoning:**", decision.reasoning)

        if decision.action != "HOLD":
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Suggested Entry", f"${decision.suggested_entry_price:.2f}")
            dc2.metric("Stop Loss", f"${decision.stop_loss:.2f}")
            dc3.metric("Take Profit", f"${decision.take_profit:.2f}")

        st.divider()

        # ── Indicators table ─────────────────────────────────────────────────
        st.subheader("📐 Technical Indicators")
        ind_data = {
            "Indicator": [
                "Price", "SMA 20", "SMA 50", "EMA 12", "EMA 26",
                "RSI 14", "MACD", "BB Upper", "BB Lower", "ATR 14",
                "Volume", "Avg Volume",
            ],
            "Value": [
                f"${snap.current_price:.2f}",
                f"${snap.sma_20:.2f}", f"${snap.sma_50:.2f}",
                f"${snap.ema_12:.2f}", f"${snap.ema_26:.2f}",
                f"{snap.rsi_14:.1f}", f"{snap.macd_histogram:+.4f}",
                f"${snap.bb_upper:.2f}", f"${snap.bb_lower:.2f}",
                f"{snap.atr_14:.2f}",
                f"{snap.volume_current:,}", f"{snap.volume_avg_20:,.0f}",
            ],
            "Signal": [
                snap.trend_label, snap.trend_label, snap.trend_label,
                snap.macd_label, snap.macd_label,
                snap.rsi_label, snap.macd_label,
                snap.bb_label, snap.bb_label,
                "", snap.volume_label, snap.volume_label,
            ],
        }
        st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)

        # ── Price chart with BBands ──────────────────────────────────────────
        st.subheader("📈 Price Chart (last 60 bars)")
        chart_df = df.tail(60)
        fig3 = go.Figure()
        fig3.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="Price",
            increasing_line_color="#00D4AA",
            decreasing_line_color="#FF4B4B",
        ))
        # Bollinger Bands overlay
        import ta as _ta
        close_s = df["close"]
        bb_obj = _ta.volatility.BollingerBands(close_s, window=20, window_dev=2)
        bb_up = bb_obj.bollinger_hband().tail(60)
        bb_lo = bb_obj.bollinger_lband().tail(60)
        fig3.add_trace(go.Scatter(x=chart_df.index, y=bb_up, line=dict(color="rgba(100,149,237,0.5)", width=1), name="BB Upper"))
        fig3.add_trace(go.Scatter(x=chart_df.index, y=bb_lo, line=dict(color="rgba(100,149,237,0.5)", width=1), name="BB Lower", fill="tonexty", fillcolor="rgba(100,149,237,0.05)"))
        fig3.update_layout(
            xaxis_rangeslider_visible=False,
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.header("💼 Portfolio")

    pf_col1, pf_col2 = st.columns([3, 1])
    pf_days = pf_col2.number_input("Trade history (days)", min_value=7, max_value=180, value=30)
    load_pf = pf_col1.button("🔄 Load Portfolio", type="primary", use_container_width=True)

    if load_pf:
        cfg = require_config()

        with st.spinner("Fetching account…"):
            try:
                account = fetch_account(cfg)
            except Exception as e:
                st.error(f"Account fetch failed: {e}")
                st.stop()

        # Account summary
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Portfolio Value", f"${account['portfolio_value']:,.2f}")
        ac2.metric("Equity", f"${account['equity']:,.2f}")
        ac3.metric("Buying Power", f"${account['buying_power']:,.2f}")
        ac4.metric("Cash", f"${account['cash']:,.2f}")

        st.divider()

        # Open positions
        with st.spinner("Fetching positions…"):
            positions = fetch_positions(cfg)

        st.subheader(f"Open Positions ({len(positions)})")
        if positions:
            pos_rows = []
            for p in positions:
                plpc = p["unrealized_plpc"] * 100
                pos_rows.append({
                    "Symbol": p["symbol"],
                    "Side": p["side"],
                    "Qty": p["qty"],
                    "Avg Entry": f"${p['avg_entry']:.2f}",
                    "Current": f"${p['current_price']:.2f}",
                    "Mkt Value": f"${p['market_value']:,.2f}",
                    "Unr. P&L": f"{p['unrealized_pl']:+.2f}",
                    "Unr. %": f"{plpc:+.1f}%",
                })
            pos_df = pd.DataFrame(pos_rows)

            def color_pl_cell(val):
                if isinstance(val, str) and val.startswith(("+", "-")):
                    return f"color: {'#00D4AA' if val.startswith('+') else '#FF4B4B'}"
                return ""

            st.dataframe(
                pos_df.style.applymap(color_pl_cell, subset=["Unr. P&L", "Unr. %"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No open positions.")

        st.divider()

        # Trade history
        with st.spinner(f"Fetching {pf_days}-day trade history…"):
            history = fetch_trade_history(cfg, days=pf_days)

        st.subheader(f"Trade History — Last {pf_days} Days")
        if history:
            hist_rows = []
            for h in history:
                hist_rows.append({
                    "Symbol": h["symbol"],
                    "Buy Qty": h["buy_qty"],
                    "Avg Buy": f"${h['avg_buy_price']:.2f}" if h["buy_qty"] else "—",
                    "Sell Qty": h["sell_qty"],
                    "Avg Sell": f"${h['avg_sell_price']:.2f}" if h["sell_qty"] else "—",
                    "Matched": h["matched_qty"],
                    "Realized P&L": f"${h['realized_pl']:+.2f}",
                    "P&L %": f"{h['realized_plpc']*100:+.1f}%" if h["matched_qty"] else "—",
                    "Open Qty": h["open_qty"],
                })
            hist_df = pd.DataFrame(hist_rows)
            total_pl = sum(h["realized_pl"] for h in history)
            st.dataframe(
                hist_df.style.applymap(color_pl_cell, subset=["Realized P&L", "P&L %"]),
                use_container_width=True,
                hide_index=True,
            )
            pl_color = "#00D4AA" if total_pl >= 0 else "#FF4B4B"
            st.markdown(
                f"**Total Realized P&L ({pf_days}d):** "
                f"<span style='color:{pl_color};font-weight:bold;'>${total_pl:+,.2f}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No closed trades in this period.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI REVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_review:
    st.header("🤖 AI Position Review")
    st.caption("Asks Claude to evaluate each open position and give a BUY / HOLD / SELL verdict.")

    run_review = st.button("🤖 Run AI Review", type="primary", use_container_width=True)

    if run_review:
        cfg = require_config()

        with st.spinner("Fetching open positions…"):
            positions = fetch_positions(cfg)

        if not positions:
            st.info("No open positions to review.")
        else:
            for pos in positions:
                sym = pos["symbol"]
                with st.spinner(f"Analysing {sym}…"):
                    try:
                        df = fetch_bars(cfg, sym, days=100)
                        snap = compute_indicators(df, symbol=sym)
                        decision = get_analysis(cfg, snap)
                    except Exception as e:
                        with st.expander(f"❌ {sym} — error"):
                            st.error(str(e))
                        continue

                action_colors = {"BUY": "#00D4AA", "SELL": "#FF4B4B", "HOLD": "#FFA500"}
                color = action_colors.get(decision.action, "#888")
                label = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(decision.action, "⚪")

                with st.expander(
                    f"{label} {sym}  —  {decision.action}  "
                    f"({decision.confidence_score:.0%} confidence)  "
                    f"|  Unr. P&L: ${pos['unrealized_pl']:+.2f} ({pos['unrealized_plpc']*100:+.1f}%)",
                    expanded=decision.action != "HOLD",
                ):
                    rv1, rv2, rv3, rv4 = st.columns(4)
                    rv1.metric("Current Price", f"${snap.current_price:.2f}")
                    rv2.metric("RSI 14", f"{snap.rsi_14:.1f}", help=snap.rsi_label)
                    rv3.metric("Trend", snap.trend_label)
                    rv4.metric("Volume", snap.volume_label)

                    st.markdown(
                        f"<div style='border-left:4px solid {color};padding-left:12px;'>"
                        f"<strong>Claude:</strong> {decision.reasoning}</div>",
                        unsafe_allow_html=True,
                    )

                    if decision.action != "HOLD":
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Suggested Entry", f"${decision.suggested_entry_price:.2f}")
                        sc2.metric("Stop Loss", f"${decision.stop_loss:.2f}")
                        sc3.metric("Take Profit", f"${decision.take_profit:.2f}")
