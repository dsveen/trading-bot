from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from trading_bot.models import IndicatorSnapshot, TradeDecision

console = Console()


def log_step(msg: str) -> None:
    """Print a timestamped narration line so the user sees what's happening."""
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{ts}[/dim] {msg}")


def show_indicators(snapshot: IndicatorSnapshot) -> None:
    table = Table(title=f"Technical Indicators - {snapshot.symbol}")
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Signal", style="bold")

    table.add_row("Price", f"${snapshot.current_price:.2f}", "")
    table.add_row("SMA 20", f"${snapshot.sma_20:.2f}", "")
    table.add_row("SMA 50", f"${snapshot.sma_50:.2f}", snapshot.trend_label)
    table.add_row("EMA 12", f"${snapshot.ema_12:.2f}", "")
    table.add_row("EMA 26", f"${snapshot.ema_26:.2f}", snapshot.macd_label)
    table.add_row("RSI (14)", f"{snapshot.rsi_14:.2f}", snapshot.rsi_label)
    table.add_row("MACD Line", f"{snapshot.macd_line:.4f}", "")
    table.add_row("MACD Signal", f"{snapshot.macd_signal:.4f}", "")
    table.add_row("MACD Hist", f"{snapshot.macd_histogram:.4f}", "")
    table.add_row("BB Upper", f"${snapshot.bb_upper:.2f}", "")
    table.add_row("BB Middle", f"${snapshot.bb_middle:.2f}", snapshot.bb_label)
    table.add_row("BB Lower", f"${snapshot.bb_lower:.2f}", "")
    table.add_row("ATR (14)", f"${snapshot.atr_14:.2f}", "")
    table.add_row("Volume", f"{snapshot.volume_current:,}", snapshot.volume_label)
    table.add_row("Vol Avg 20", f"{snapshot.volume_avg_20:,.0f}", "")

    console.print(table)


def show_indicators_compact(snapshot: IndicatorSnapshot) -> None:
    """One-line summary for watch mode — less screen clutter."""
    trend_color = {"Bullish": "green", "Bearish": "red"}.get(snapshot.trend_label, "yellow")
    rsi_color = {"OVERBOUGHT": "red", "OVERSOLD": "green"}.get(snapshot.rsi_label, "white")
    vol_icon = "+" if snapshot.volume_label == "Above Average" else "-"

    console.print(
        f"  [cyan]{snapshot.symbol}[/cyan] "
        f"${snapshot.current_price:.2f}  "
        f"RSI [{ rsi_color}]{snapshot.rsi_14:.1f}[/{rsi_color}]  "
        f"Trend [{trend_color}]{snapshot.trend_label}[/{trend_color}]  "
        f"MACD {snapshot.macd_histogram:+.4f}  "
        f"BB {snapshot.bb_label}  "
        f"Vol {vol_icon}{snapshot.volume_current:,}"
    )


def show_decision(decision: TradeDecision) -> None:
    action_colors = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
    color = action_colors.get(decision.action.value, "white")

    panel_content = (
        f"[bold {color}]{decision.action.value}[/bold {color}]\n\n"
        f"[bold]Reasoning:[/bold] {decision.reasoning}\n\n"
        f"[bold]Confidence:[/bold] {decision.confidence_score:.0%}\n"
        f"[bold]Shares:[/bold] {decision.suggested_shares}\n"
        f"[bold]Entry:[/bold] ${decision.suggested_entry_price:.2f}\n"
        f"[bold]Stop Loss:[/bold] ${decision.stop_loss:.2f}\n"
        f"[bold]Take Profit:[/bold] ${decision.take_profit:.2f}\n"
        f"[bold]Risk/Reward:[/bold] {decision.risk_reward_ratio:.2f}:1\n"
        f"[bold]Horizon:[/bold] {decision.time_horizon}"
    )

    console.print(Panel(panel_content, title="Claude's Recommendation", border_style=color))


def show_decision_compact(decision: TradeDecision) -> None:
    """Compact decision output for watch mode."""
    action_colors = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
    color = action_colors.get(decision.action.value, "white")

    console.print(
        f"  [{color}]{decision.action.value}[/{color}] "
        f"({decision.confidence_score:.0%}) "
        f"Entry ${decision.suggested_entry_price:.2f}  "
        f"SL ${decision.stop_loss:.2f}  "
        f"TP ${decision.take_profit:.2f}  "
        f"R:R {decision.risk_reward_ratio:.1f}:1  "
        f"Shares {decision.suggested_shares}"
    )
    console.print(f"  [dim]Reasoning: {decision.reasoning}[/dim]")


def show_watch_header(symbol: str, interval: int, model: str) -> None:
    console.print(Rule(f"Watching {symbol}  |  every {interval}s  |  model: {model}"))
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")


def show_portfolio(positions: list[dict]) -> None:
    if not positions:
        console.print("[yellow]No open positions.[/yellow]")
        return

    table = Table(title="Portfolio Positions")
    table.add_column("Symbol", style="cyan")
    table.add_column("Qty", justify="right")
    table.add_column("Avg Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Mkt Value", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")

    for p in positions:
        pl_color = "green" if p["unrealized_pl"] >= 0 else "red"
        table.add_row(
            p["symbol"],
            f"{p['qty']:.0f}",
            f"${p['avg_entry']:.2f}",
            f"${p['current_price']:.2f}",
            f"${p['market_value']:.2f}",
            f"[{pl_color}]${p['unrealized_pl']:.2f}[/{pl_color}]",
            f"[{pl_color}]{p['unrealized_plpc']:.2%}[/{pl_color}]",
        )

    console.print(table)


def show_position_review(
    pos: dict,
    snapshot: IndicatorSnapshot,
    decision: TradeDecision | None,
    underlying: str,
    is_option: bool,
) -> None:
    """Render an AI-review panel for one open position."""
    symbol = pos["symbol"]
    side = pos["side"]
    qty = pos["qty"]
    avg_entry = pos["avg_entry"]
    current = pos["current_price"]
    mkt_val = pos["market_value"]
    pl = pos["unrealized_pl"]
    plpc = pos["unrealized_plpc"]

    pl_color = "green" if pl >= 0 else "red"
    side_color = "green" if side == "long" else "red"
    action = decision.action.value if decision else "N/A"
    action_colors = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
    border = action_colors.get(action, "dim")

    qty_label = f"{qty:.0f} contract(s)" if is_option else f"{qty:.0f} share(s)"
    title = (
        f"[bold]{symbol}[/bold]"
        + (f"  [dim]option on {underlying}[/dim]" if is_option else "")
        + f"  [{side_color}]{side.upper()} {qty_label}[/{side_color}]"
        + f"  [{pl_color}]{pl:+,.2f} ({plpc:+.1%})[/{pl_color}]"
    )

    lines = []

    # Position details
    move = current - avg_entry
    move_color = "green" if move >= 0 else "red"
    lines.append(
        f"  Entry [bold]${avg_entry:.2f}[/bold]  →  "
        f"Now [bold]${current:.2f}[/bold]  "
        f"([{move_color}]{move:+.2f}[/{move_color}])  │  "
        f"Market value [bold]${mkt_val:,.2f}[/bold]"
    )

    lines.append("")
    lines.append("  [bold dim]── INDICATORS ──────────────────────────────[/bold dim]")

    rsi_color = "red" if snapshot.rsi_14 > 70 else "green" if snapshot.rsi_14 < 30 else "white"
    lines.append(
        f"  RSI    [{rsi_color}]{snapshot.rsi_14:.1f}[/{rsi_color}]"
        f"  {snapshot.rsi_label}"
    )

    macd_color = "green" if snapshot.macd_histogram > 0 else "red"
    lines.append(
        f"  MACD   [{macd_color}]{snapshot.macd_label}[/{macd_color}]"
        f"  hist [{macd_color}]{snapshot.macd_histogram:+.4f}[/{macd_color}]"
    )

    lines.append(f"  BB     {snapshot.bb_label}")

    trend_color = {"Bullish": "green", "Bearish": "red"}.get(snapshot.trend_label, "yellow")
    lines.append(f"  Trend  [{trend_color}]{snapshot.trend_label}[/{trend_color}]")

    vol_color = "green" if snapshot.volume_label == "Above Average" else "dim"
    lines.append(
        f"  ATR    ${snapshot.atr_14:.2f}  ·  "
        f"Vol [{vol_color}]{snapshot.volume_label}[/{vol_color}]"
    )

    lines.append("")
    lines.append("  [bold dim]── CLAUDE'S VERDICT ────────────────────────[/bold dim]")

    if decision:
        lines.append(
            f"  [bold {border}]{action}[/bold {border}]"
            f"  {decision.confidence_score:.0%} confidence"
            f"  [dim]{decision.time_horizon}[/dim]"
        )
        lines.append(f'  [italic]"{decision.reasoning}"[/italic]')
        lines.append("")
        lines.append(
            f"  [dim]Entry ${decision.suggested_entry_price:.2f}  "
            f"Stop ${decision.stop_loss:.2f}  "
            f"Target ${decision.take_profit:.2f}  "
            f"R/R {decision.risk_reward_ratio:.1f}:1  "
            f"Shares {decision.suggested_shares}[/dim]"
        )
    else:
        lines.append("  [red]Claude analysis unavailable.[/red]")

    console.print(Panel("\n".join(lines), title=title, border_style=border, padding=(0, 1)))
    console.print()


def show_trade_history(trades: list[dict], days: int) -> None:
    """Show a table of completed (and partial) trades with realized P&L."""
    if not trades:
        console.print(f"[yellow]No filled orders found in the last {days} days.[/yellow]")
        return

    table = Table(title=f"Trade History — last {days} days")
    table.add_column("Symbol", style="cyan")
    table.add_column("Bought Qty", justify="right")
    table.add_column("Avg Buy", justify="right")
    table.add_column("Sold Qty", justify="right")
    table.add_column("Avg Sell", justify="right")
    table.add_column("Matched", justify="right")
    table.add_column("Realized P&L", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("Open Qty", justify="right")

    total_pl = 0.0
    for t in trades:
        pl = t["realized_pl"]
        total_pl += pl
        pl_color = "green" if pl > 0 else "red" if pl < 0 else "white"
        open_qty = t["open_qty"]
        open_color = "cyan" if open_qty > 0 else "dim"

        avg_buy_str = f"${t['avg_buy_price']:.2f}" if t["buy_qty"] > 0 else "—"
        avg_sell_str = f"${t['avg_sell_price']:.2f}" if t["sell_qty"] > 0 else "—"
        pl_str = f"[{pl_color}]${pl:+,.2f}[/{pl_color}]" if t["matched_qty"] > 0 else "—"
        plpc_str = (
            f"[{pl_color}]{t['realized_plpc']:+.2%}[/{pl_color}]"
            if t["matched_qty"] > 0 else "—"
        )

        table.add_row(
            t["symbol"],
            f"{t['buy_qty']:.0f}" if t["buy_qty"] > 0 else "—",
            avg_buy_str,
            f"{t['sell_qty']:.0f}" if t["sell_qty"] > 0 else "—",
            avg_sell_str,
            f"{t['matched_qty']:.0f}" if t["matched_qty"] > 0 else "—",
            pl_str,
            plpc_str,
            f"[{open_color}]{open_qty:+.0f}[/{open_color}]" if open_qty != 0 else "0",
        )

    console.print(table)
    total_color = "green" if total_pl > 0 else "red" if total_pl < 0 else "white"
    console.print(
        f"  Total realized P&L: [{total_color}]${total_pl:+,.2f}[/{total_color}]"
    )


def show_account(account: dict, mode: str) -> None:
    mode_color = "yellow" if mode == "paper" else "red"
    table = Table(title="Account Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Mode", f"[{mode_color}]{mode.upper()}[/{mode_color}]")
    table.add_row("Equity", f"${account['equity']:,.2f}")
    table.add_row("Buying Power", f"${account['buying_power']:,.2f}")
    table.add_row("Cash", f"${account['cash']:,.2f}")
    table.add_row("Portfolio Value", f"${account['portfolio_value']:,.2f}")
    table.add_row("Day Trades", str(account["day_trade_count"]))

    console.print(table)
