import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import click
from rich.console import Console

from trading_bot.config import load_config
from trading_bot.market_data import (
    fetch_bars,
    fetch_minute_bars,
    fetch_intraday_bars,
    fetch_latest_price,
    fetch_account,
    fetch_positions,
    fetch_trade_history,
    get_position,
    find_best_option,
    find_otm_call,
    get_market_clock,
    check_0dte_available,
)
from trading_bot.indicators import compute_indicators
from trading_bot.analyst import get_analysis, get_scalp_analysis
from trading_bot.signals import scan_entry, PositionTracker
from trading_bot.safety import run_safety_checks
from trading_bot.trader import (
    execute_bracket_order,
    execute_market_order,
    close_position,
    execute_option_buy,
    close_option_position,
)
from trading_bot.display import (
    console,
    log_step,
    show_indicators,
    show_indicators_compact,
    show_decision,
    show_decision_compact,
    show_watch_header,
    show_portfolio,
    show_trade_history,
    show_account,
    show_position_review,
)
from rich.rule import Rule


@click.group()
def cli():
    """AI-powered trading bot using Claude and Alpaca."""
    pass


def _check_market_open(config) -> bool:
    """Wait until the US market is open before returning True.

    If the market is already open, returns immediately. If closed, sleeps in
    30s increments and prints a countdown until it opens. Returns False only
    if the user presses Ctrl+C while waiting.
    """
    try:
        clock = get_market_clock(config)
    except Exception as e:
        log_step(f"[yellow]Could not fetch market clock: {e} — proceeding anyway.[/yellow]")
        return True

    if clock["is_open"]:
        close_str = clock["next_close"].strftime("%I:%M %p ET")
        log_step(f"Market is [bold green]OPEN[/bold green] — closes at {close_str}")
        return True

    # Market is closed — compute time until next open
    now = clock["timestamp"]
    next_open = clock["next_open"]
    delta = next_open - now
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    open_str = next_open.strftime("%A %b %-d at %-I:%M %p ET")

    log_step(
        f"Market is [bold red]CLOSED[/bold red] — "
        f"next open: {open_str}  ({hours}h {minutes}m from now)"
    )
    console.print(f"[dim]Waiting for market open... Ctrl+C to cancel.[/dim]")

    try:
        while True:
            time.sleep(30)
            clock = get_market_clock(config)
            if clock["is_open"]:
                log_step("[bold green]Market is now OPEN — starting strategy.[/bold green]")
                return True
            delta = clock["next_open"] - clock["timestamp"]
            h, rem = divmod(int(delta.total_seconds()), 3600)
            m = rem // 60
            log_step(f"[dim]Still waiting... {h}h {m}m until open[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Wait cancelled.[/yellow]")
        return False


def _run_analysis(config, symbol: str, compact: bool = False):
    """Single analysis pass with step-by-step narration (daily bars)."""
    log_step(f"Fetching daily bars for [cyan]{symbol}[/cyan]...")
    df = fetch_bars(config, symbol)
    log_step(f"Got {len(df)} bars. Latest close: ${float(df['close'].iloc[-1]):.2f}")

    log_step("Computing indicators (standard windows)...")
    snapshot = compute_indicators(df, symbol, fast=False)

    if compact:
        show_indicators_compact(snapshot)
    else:
        show_indicators(snapshot)

    log_step(f"Sending to Claude ({config.claude_model})...")
    t0 = time.time()
    decision = get_analysis(config, snapshot)
    elapsed = time.time() - t0
    log_step(f"Claude responded in {elapsed:.1f}s")

    if compact:
        show_decision_compact(decision)
    else:
        show_decision(decision)

    return decision


@cli.command()
@click.argument("symbol")
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
@click.option("--watch", type=int, default=None, metavar="SECS",
              help="Re-analyze every N seconds until Ctrl+C.")
def analyze(symbol: str, paper: bool, watch: int | None):
    """Fetch data, compute indicators, and get Claude's recommendation."""
    symbol = symbol.upper()
    config = load_config()
    if not paper:
        config.trading_mode = "live"

    if watch is None:
        _run_analysis(config, symbol, compact=False)
        return

    show_watch_header(symbol, watch, config.claude_model)
    iteration = 0
    try:
        while True:
            iteration += 1
            console.print(f"\n[bold]--- Iteration {iteration} ---[/bold]")
            try:
                _run_analysis(config, symbol, compact=True)
            except Exception as e:
                log_step(f"[red]Error: {e}[/red]")
            log_step(f"Sleeping {watch}s...")
            time.sleep(watch)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Stopped after {iteration} iterations.[/yellow]")


@cli.command()
@click.argument("symbol")
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
@click.option("--interval", type=float, default=2,
              help="Seconds between each tick (default 2).")
@click.option("--refresh", type=int, default=30,
              help="Seconds between full bar refresh + signal scan (default 30).")
@click.option("--shares", type=int, default=None,
              help="Override share count (default: Claude decides).")
@click.option("--options", "use_options", is_flag=True, default=False,
              help="Trade options (calls/puts) instead of stock shares.")
@click.option("--contracts", type=int, default=1,
              help="Number of option contracts to buy (default 1, each = 100 shares).")
@click.option("--dte", type=int, default=7,
              help="Max days-to-expiry when selecting option contracts (default 7).")
def scalp(
    symbol: str,
    paper: bool,
    interval: float,
    refresh: int,
    shares: int | None,
    use_options: bool,
    contracts: int,
    dte: int,
):
    """High-frequency scalping mode with signal filtering.

    Fast loop (every 2s): checks latest price, manages exits mechanically.
    Full scan (every 30s): fetches bars, computes indicators, scans for entries.
    Claude is only called when the code scanner finds a setup (3+ signals).

    With --options: buys call/put contracts instead of shares. Each contract
    controls 100 shares. Exits based on underlying stock price movement.
    """
    symbol = symbol.upper()
    config = load_config()
    if not paper:
        config.trading_mode = "live"
    config.auto_execute = True

    if not _check_market_open(config):
        return

    mode_tag = "[yellow]PAPER[/yellow]" if config.is_paper else "[red bold]LIVE[/red bold]"
    instrument = "[magenta]OPTIONS[/magenta]" if use_options else "stocks"
    console.print(f"\n[bold red]SCALP MODE[/bold red]  {mode_tag}  "
                  f"[cyan]{symbol}[/cyan]  {instrument}  tick {interval}s  refresh {refresh}s  "
                  f"model: {config.claude_model}")
    if use_options:
        console.print(f"[dim]Options: {contracts} contract(s), max {dte} DTE. "
                      f"Calls on bullish, puts on bearish.[/dim]")
    console.print("[dim]Fast exits, filtered entries. Ctrl+C to stop.[/dim]\n")

    tracker = PositionTracker()
    iteration = 0
    trades_executed = 0
    wins = 0
    losses = 0
    total_pl = 0.0
    ai_calls = 0
    last_refresh = 0.0  # force immediate first refresh
    cached_snapshot = None
    active_option_symbol: str | None = None  # tracks current option contract

    try:
        while True:
            iteration += 1
            now = time.time()
            is_refresh_tick = (now - last_refresh) >= refresh

            # --- FAST TICK: just get latest price ---
            try:
                price = fetch_latest_price(config, symbol)
            except Exception as e:
                log_step(f"[red]Price fetch error: {e}[/red]")
                time.sleep(interval)
                continue

            # --- EXIT CHECK (every tick, no API cost) ---
            if tracker.is_active:
                should_exit, exit_reason = tracker.check_exit(price)
                if should_exit:
                    log_step(f"[bold yellow]EXIT @ ${price:.2f}: {exit_reason}[/bold yellow]")
                    try:
                        if use_options and active_option_symbol:
                            position = get_position(config, active_option_symbol)
                            close_option_position(config, active_option_symbol)
                            realized_pl = position["unrealized_pl"] if position else 0.0
                            log_step(f"[dim]Closed option {active_option_symbol}[/dim]")
                            active_option_symbol = None
                        else:
                            position = get_position(config, symbol)
                            close_position(config, symbol)
                            realized_pl = position["unrealized_pl"] if position else 0.0
                        trades_executed += 1
                        total_pl += realized_pl
                        if realized_pl >= 0:
                            wins += 1
                        else:
                            losses += 1
                        tracker.clear()
                        color = "green" if realized_pl >= 0 else "red"
                        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                        log_step(
                            f"[{color}]Closed! P&L: ${realized_pl:.2f}  "
                            f"Total: ${total_pl:.2f}  "
                            f"W/L: {wins}/{losses} ({win_rate:.0f}%)  "
                            f"AI calls: {ai_calls}[/{color}]"
                        )
                    except Exception as e:
                        log_step(f"[red]Close error: {e}[/red]")
                    time.sleep(interval)
                    continue
                else:
                    # Quiet hold — just show price on refresh ticks
                    if is_refresh_tick:
                        pos_symbol = active_option_symbol if (use_options and active_option_symbol) else symbol
                        position = get_position(config, pos_symbol)
                        if position:
                            pl = position["unrealized_pl"]
                            color = "green" if pl >= 0 else "red"
                            hold_label = f"opt {active_option_symbol}" if use_options and active_option_symbol else "stock"
                            log_step(
                                f"#{iteration} ${price:.2f}  "
                                f"[{color}]P&L ${pl:.2f}[/{color}]  "
                                f"peak ${tracker.highest_since_entry:.2f}  "
                                f"holding {hold_label}..."
                            )
                        elif not position and tracker.is_active:
                            log_step("[yellow]Position gone — resetting tracker.[/yellow]")
                            tracker.clear()
                            active_option_symbol = None
                        last_refresh = now
                    time.sleep(interval)
                    continue

            # --- FULL SCAN (only on refresh ticks, when flat) ---
            if not is_refresh_tick:
                time.sleep(interval)
                continue

            last_refresh = now
            log_step(f"#{iteration} [cyan]{symbol}[/cyan] ${price:.2f} — full scan...")

            try:
                df = fetch_minute_bars(config, symbol, minutes=120)
            except Exception as e:
                log_step(f"[red]Data error: {e}[/red]")
                time.sleep(interval)
                continue

            if len(df) < 15:
                log_step(f"[yellow]Only {len(df)} bars — need 15+. Market open?[/yellow]")
                time.sleep(interval)
                continue

            snapshot = compute_indicators(df, symbol, fast=True)
            cached_snapshot = snapshot
            show_indicators_compact(snapshot)

            # Scan for entry
            signal = scan_entry(snapshot)

            if signal.direction == "none":
                log_step(f"[dim]No signal ({signal.strength}/5). "
                         f"{', '.join(signal.reasons)}[/dim]")
                time.sleep(interval)
                continue

            # Signal found — call Claude
            signal_color = "green" if signal.direction == "long" else "red"
            log_step(
                f"[{signal_color} bold]SIGNAL: {signal.direction.upper()} "
                f"({signal.strength}/5)[/{signal_color} bold]  "
                + "  ".join(signal.reasons)
            )

            log_step(f"Asking Claude to confirm...")
            ai_calls += 1
            t0 = time.time()
            try:
                decision = get_scalp_analysis(
                    config, snapshot,
                    signal_direction=signal.direction,
                    signal_strength=signal.strength,
                    signal_reasons=signal.reasons,
                )
            except Exception as e:
                log_step(f"[red]Claude error: {e}[/red]")
                time.sleep(interval)
                continue
            elapsed = time.time() - t0
            log_step(f"Claude: {elapsed:.1f}s")
            show_decision_compact(decision)

            if shares is not None:
                decision.suggested_shares = shares

            if decision.action.value == "HOLD":
                log_step("[yellow]Claude rejected — skipping.[/yellow]")
            elif decision.action.value in ("BUY", "SELL"):
                side = "long" if decision.action.value == "BUY" else "short"

                if use_options:
                    # --- OPTIONS ENTRY ---
                    log_step(f"[magenta]Looking for {side.upper()} option ({dte}d DTE)...[/magenta]")
                    try:
                        opt = find_best_option(
                            config, symbol, side, price,
                            min_dte=1, max_dte=dte,
                        )
                    except Exception as e:
                        log_step(f"[red]Option lookup error: {e}[/red]")
                        time.sleep(interval)
                        continue

                    if opt is None:
                        log_step(f"[yellow]No option contract found (±3% of ${price:.2f}, ≤{dte}d DTE). Skipping.[/yellow]")
                        time.sleep(interval)
                        continue

                    opt_type_color = "green" if opt["type"] == "CALL" else "red"
                    log_step(
                        f"[{opt_type_color}]Contract: {opt['symbol']}  "
                        f"{opt['type']} ${opt['strike']:.2f}  "
                        f"exp {opt['expiry']} ({opt['dte']}d DTE)[/{opt_type_color}]"
                    )
                    try:
                        execute_option_buy(config, opt["symbol"], contracts)
                        active_option_symbol = opt["symbol"]
                        tracker.enter(price=price, side=side, atr=snapshot.atr_14)
                        sl = price - snapshot.atr_14 if side == "long" else price + snapshot.atr_14
                        tp = price + 2 * snapshot.atr_14 if side == "long" else price - 2 * snapshot.atr_14
                        log_step(
                            f"[green]Option bought! {contracts}x {opt['symbol']}  "
                            f"Underlying SL ${sl:.2f}  TP ${tp:.2f}[/green]"
                        )
                    except Exception as e:
                        log_step(f"[red]Option order error: {e}[/red]")
                else:
                    # --- STOCK ENTRY ---
                    checks = run_safety_checks(config, decision)
                    failed = [c for c in checks if not c.passed]
                    if failed:
                        for c in failed:
                            log_step(f"[red]BLOCKED: {c.reason}[/red]")
                    else:
                        log_step(
                            f"[bold]{decision.action.value} "
                            f"{decision.suggested_shares} @ market...[/bold]"
                        )
                        try:
                            execute_market_order(config, symbol, decision)
                            tracker.enter(price=price, side=side, atr=snapshot.atr_14)
                            sl = price - snapshot.atr_14 if side == "long" else price + snapshot.atr_14
                            tp = price + 2 * snapshot.atr_14 if side == "long" else price - 2 * snapshot.atr_14
                            log_step(
                                f"[green]Filled! SL ${sl:.2f}  TP ${tp:.2f}  "
                                f"Trail {tracker.trailing_stop_atr}x ATR[/green]"
                            )
                        except Exception as e:
                            log_step(f"[red]Order error: {e}[/red]")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Scalp session ended.[/yellow]")
        console.print(f"  Ticks: {iteration}  AI calls: {ai_calls} (saved {iteration - ai_calls})")
        mode_label = "options" if use_options else "stock"
        console.print(f"  Trades: {trades_executed} ({mode_label})  W/L: {wins}/{losses}")
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        console.print(f"  Win rate: {win_rate:.0f}%")
        color = "green" if total_pl >= 0 else "red"
        console.print(f"  Realized P&L: [{color}]${total_pl:.2f}[/{color}]")

        # Check for unclosed positions
        if use_options and active_option_symbol:
            pos = get_position(config, active_option_symbol)
            if pos:
                console.print(
                    f"  [yellow]WARNING: Open option position {active_option_symbol} "
                    f"({int(pos['qty'])} contracts) — close manually.[/yellow]"
                )
        else:
            pos = get_position(config, symbol)
            if pos:
                console.print(
                    f"  [yellow]WARNING: Open {pos['side']} position "
                    f"{int(pos['qty'])} shares — close manually.[/yellow]"
                )


@cli.command()
@click.argument("symbol")
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
@click.option("--auto", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.option("--shares", type=int, default=None, help="Override suggested shares.")
def trade(symbol: str, paper: bool, auto: bool, shares: int | None):
    """Analyze a stock and execute a trade."""
    symbol = symbol.upper()
    config = load_config()
    if not paper:
        config.trading_mode = "live"
    if auto:
        config.auto_execute = True

    if not _check_market_open(config):
        return

    decision = _run_analysis(config, symbol, compact=False)

    if shares is not None:
        decision.suggested_shares = shares

    checks = run_safety_checks(config, decision)
    failed = [c for c in checks if not c.passed]
    if failed:
        for c in failed:
            console.print(f"[red]BLOCKED:[/red] {c.reason}")
        return

    if decision.action.value == "HOLD":
        console.print("[yellow]Decision is HOLD - no trade to execute.[/yellow]")
        return

    if not config.auto_execute:
        mode_label = "[yellow]PAPER[/yellow]" if config.is_paper else "[red]LIVE[/red]"
        console.print(f"\nMode: {mode_label}")
        if not click.confirm("Execute this trade?"):
            console.print("[yellow]Trade cancelled.[/yellow]")
            return

    log_step("Submitting bracket order to Alpaca...")
    result = execute_bracket_order(config, symbol, decision)

    if result["status"] == "submitted":
        console.print(f"[green]Order submitted![/green] ID: {result['order_id']}")
        console.print(
            f"  {result['side']} {result['qty']} {result['symbol']} "
            f"@ ${result['limit_price']:.2f} "
            f"(SL: ${result['stop_loss']:.2f}, TP: ${result['take_profit']:.2f})"
        )
    else:
        console.print(f"[red]Order failed:[/red] {result.get('reason', 'unknown')}")


@cli.command()
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
@click.option("--days", type=int, default=30,
              help="Days of trade history to fetch (default 30).")
def portfolio(paper: bool, days: int):
    """Show current positions and trade history with realized P&L."""
    config = load_config()
    if not paper:
        config.trading_mode = "live"

    log_step("Fetching open positions from Alpaca...")
    positions = fetch_positions(config)
    show_portfolio(positions)

    console.print()
    log_step(f"Fetching trade history (last {days} days)...")
    trades = fetch_trade_history(config, days=days)
    show_trade_history(trades, days=days)


def _get_underlying(symbol: str) -> str:
    """Return the underlying ticker from an OCC option symbol.

    OCC format: AAPL241220C00170000 — ticker is every char before the first digit.
    For plain stock symbols (no digits) the input is returned unchanged.
    """
    for i, c in enumerate(symbol):
        if c.isdigit():
            return symbol[:i] if i > 0 else symbol
    return symbol


@cli.command("review")
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
@click.option("--days", type=int, default=60,
              help="Days of price history to fetch per symbol (default 60).")
def review(paper: bool, days: int):
    """AI dashboard of every open position — runs once and exits.

    \b
    For each position Claude assesses the current setup and recommends
    whether to HOLD, add more (BUY), or close (SELL).
    Option positions are analysed via their underlying stock.
    """
    config = load_config()
    if not paper:
        config.trading_mode = "live"

    log_step("Fetching open positions...")
    positions = fetch_positions(config)

    if not positions:
        console.print("[yellow]No open positions.[/yellow]")
        return

    log_step("Fetching account summary...")
    account = fetch_account(config)

    mode_tag = "[yellow]PAPER[/yellow]" if config.is_paper else "[red bold]LIVE[/red bold]"
    total_pl = sum(p["unrealized_pl"] for p in positions)
    total_plpc = total_pl / account["portfolio_value"] if account["portfolio_value"] else 0
    total_color = "green" if total_pl >= 0 else "red"

    console.print()
    console.print(Rule(
        f"PORTFOLIO REVIEW  {mode_tag}  "
        f"{len(positions)} position(s)  model: {config.claude_model}"
    ))
    console.print(
        f"  Portfolio [bold]${account['portfolio_value']:,.2f}[/bold]  "
        f"Buying power ${account['buying_power']:,.2f}  "
        f"Open P&L [{total_color}]{total_pl:+,.2f} ({total_plpc:+.1%})[/{total_color}]\n"
    )

    errors = 0
    for i, pos in enumerate(positions, 1):
        symbol = pos["symbol"]
        underlying = _get_underlying(symbol)
        is_option = underlying != symbol

        label = f"{symbol}" + (f" (option → {underlying})" if is_option else "")
        log_step(f"[{i}/{len(positions)}] {label} — fetching {days}d of bars...")

        try:
            df = fetch_bars(config, underlying, days=days)
        except Exception as e:
            console.print(f"  [red]Data error for {underlying}: {e}[/red]")
            errors += 1
            continue

        try:
            snapshot = compute_indicators(df, underlying, fast=False)
        except Exception as e:
            console.print(f"  [red]Indicator error for {underlying}: {e}[/red]")
            errors += 1
            continue

        log_step(f"  Asking Claude ({config.claude_model})...")
        t0 = time.time()
        try:
            decision = get_analysis(config, snapshot)
            log_step(f"  Response: {time.time() - t0:.1f}s  →  {decision.action.value}")
        except Exception as e:
            log_step(f"  [red]Claude error: {e}[/red]")
            decision = None
            errors += 1

        show_position_review(pos, snapshot, decision, underlying, is_option)

    suffix = f"  ({errors} error(s))" if errors else ""
    console.print(Rule(f"Review complete — {len(positions)} position(s) analysed{suffix}"))


@cli.command()
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
def status(paper: bool):
    """Show account equity, buying power, and trading mode."""
    config = load_config()
    if not paper:
        config.trading_mode = "live"

    log_step("Fetching account info from Alpaca...")
    account = fetch_account(config)
    show_account(account, config.trading_mode)


@dataclass
class _ManagedOption:
    """Tracks per-position state for options-swing multi-position management."""
    option_symbol: str
    contracts: int
    peak_value: float = 0.0
    trailing_active: bool = False
    next_ask_threshold: float = 1.0  # start prompting at +100%
    check_num: int = 0


@cli.command("options-swing")
@click.argument("symbol")
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
@click.option("--contracts", type=int, default=1,
              help="Number of contracts per entry (default 1, each = 100 shares).")
@click.option("--auto", is_flag=True, default=False,
              help="Skip buy confirmation prompt.")
@click.option("--interval", type=int, default=60,
              help="Seconds between position checks (default 60).")
@click.option("--refresh", type=int, default=300,
              help="Seconds between new-entry scans (default 300 = 5 min).")
@click.option("--max-positions", type=int, default=3,
              help="Max simultaneous open positions (default 3).")
def options_swing(
    symbol: str,
    paper: bool,
    contracts: int,
    auto: bool,
    interval: int,
    refresh: int,
    max_positions: int,
):
    """Buy OTM calls and manage multiple positions simultaneously.

    \b
    Strategy rules:
      • Continuously scans for bullish setups (every --refresh seconds)
      • Selects CALL option: 15% OTM, 45–79 DTE (prefers 45–60)
      • Trailing stop: 5% below peak, ONLY activates after +30% gain
      • At +100%: prompts you to sell (hold or exit your choice)
      • Buys new contracts while existing positions are still open
      • Up to --max-positions simultaneous open contracts

    \b
    Example:
      trading-bot options-swing AAPL
      trading-bot options-swing AAPL --contracts 2 --max-positions 5 --refresh 120
    """
    symbol = symbol.upper()
    config = load_config()
    if not paper:
        config.trading_mode = "live"

    if not _check_market_open(config):
        return

    mode_tag = "[yellow]PAPER[/yellow]" if config.is_paper else "[red bold]LIVE[/red bold]"
    console.print(
        f"\n[bold magenta]OPTIONS SWING[/bold magenta]  {mode_tag}  "
        f"[cyan]{symbol}[/cyan]  "
        f"15% OTM CALL  45–79 DTE  "
        f"{contracts} contract(s)/entry  max {max_positions} open"
    )
    console.print(
        f"[dim]Monitor every {interval}s · scan for new entries every {refresh}s · "
        "trailing stop 5% from peak (activates at +30%) · prompt at +100%[/dim]\n"
    )

    managed: list[_ManagedOption] = []  # all currently tracked positions
    last_scan = 0.0       # force immediate first scan

    try:
        while True:
            now = time.time()
            do_scan = (now - last_scan) >= refresh

            # ── POSITION MONITORING (every interval) ──────────────────────────
            closed_symbols = []
            for pos_obj in list(managed):
                pos_obj.check_num += 1
                option_symbol = pos_obj.option_symbol

                try:
                    position = get_position(config, option_symbol)
                except Exception as e:
                    log_step(f"[red]Position fetch error ({option_symbol}): {e}[/red]")
                    continue

                if position is None:
                    console.print(
                        f"[yellow]{option_symbol} — not found "
                        "(closed, exercised, or expired).[/yellow]"
                    )
                    closed_symbols.append(option_symbol)
                    continue

                mkt_val = position["market_value"]
                plpc = position["unrealized_plpc"]
                pl_dollar = position["unrealized_pl"]

                # Update peak
                if mkt_val > pos_obj.peak_value:
                    pos_obj.peak_value = mkt_val

                # Activate trailing stop at +30%
                if not pos_obj.trailing_active and plpc >= 0.30:
                    pos_obj.trailing_active = True
                    log_step(
                        f"[bold green]{option_symbol}  "
                        f"Trailing stop ACTIVATED at {plpc:+.1%}  "
                        f"stop = ${pos_obj.peak_value * 0.95:.2f}[/bold green]"
                    )

                pl_color = "green" if pl_dollar >= 0 else "red"
                trail_tag = (
                    "  [bold cyan]TRAIL ON[/bold cyan]"
                    if pos_obj.trailing_active
                    else "  [dim]no stop yet[/dim]"
                )
                log_step(
                    f"#{pos_obj.check_num} [cyan]{option_symbol}[/cyan]  "
                    f"val ${mkt_val:.2f}  "
                    f"[{pl_color}]{plpc:+.1%} (${pl_dollar:+.2f})[/{pl_color}]  "
                    f"peak ${pos_obj.peak_value:.2f}"
                    f"{trail_tag}"
                )

                # +100% prompt (repeats at +150%, +200%, etc. if user holds)
                if plpc >= pos_obj.next_ask_threshold:
                    console.print(
                        f"\n[bold yellow]{option_symbol} is up {plpc:.0%}![/bold yellow]  "
                        f"(${pl_dollar:+.2f})"
                    )
                    if click.confirm("Sell now?"):
                        log_step(f"Closing {option_symbol}...")
                        try:
                            close_option_position(config, option_symbol)
                            console.print(
                                f"[bold green]Sold {option_symbol}!  "
                                f"~${pl_dollar:.2f} ({plpc:+.1%})[/bold green]"
                            )
                        except Exception as e:
                            console.print(f"[red]Close error: {e}[/red]")
                        closed_symbols.append(option_symbol)
                        continue
                    else:
                        next_thr = pos_obj.next_ask_threshold + 0.50
                        console.print(
                            f"[dim]Holding. Next prompt at {next_thr:.0%}.[/dim]"
                        )
                        pos_obj.next_ask_threshold = next_thr

                # Trailing stop check (only after activated)
                if pos_obj.trailing_active and pos_obj.peak_value > 0:
                    stop_level = pos_obj.peak_value * 0.95
                    if mkt_val <= stop_level:
                        drop_pct = (pos_obj.peak_value - mkt_val) / pos_obj.peak_value
                        log_step(
                            f"[bold red]TRAILING STOP HIT {option_symbol}  "
                            f"peak ${pos_obj.peak_value:.2f} → ${mkt_val:.2f}  "
                            f"(-{drop_pct:.1%})[/bold red]"
                        )
                        try:
                            close_option_position(config, option_symbol)
                            console.print(
                                f"[yellow]{option_symbol} closed.  "
                                f"P&L: ${pl_dollar:+.2f} ({plpc:+.1%})[/yellow]"
                            )
                        except Exception as e:
                            console.print(f"[red]Close error: {e}[/red]")
                        closed_symbols.append(option_symbol)

            # Remove closed positions from tracking list
            if closed_symbols:
                managed = [m for m in managed if m.option_symbol not in closed_symbols]
                console.print(
                    f"[dim]Active positions: {len(managed)}/{max_positions}[/dim]"
                )

            # ── ENTRY SCAN (every refresh seconds, if slots available) ────────
            if do_scan:
                last_scan = now
                open_count = len(managed)

                if open_count >= max_positions:
                    log_step(
                        f"[dim]At position limit ({open_count}/{max_positions}) — "
                        "skipping entry scan.[/dim]"
                    )
                else:
                    log_step(
                        f"[bold]Entry scan[/bold]  "
                        f"({open_count}/{max_positions} positions open)  "
                        f"Fetching daily bars for [cyan]{symbol}[/cyan]..."
                    )
                    try:
                        df = fetch_bars(config, symbol)
                        snapshot = compute_indicators(df, symbol, fast=False)
                        show_indicators_compact(snapshot)

                        log_step(f"Asking Claude ({config.claude_model})...")
                        t0 = time.time()
                        decision = get_analysis(config, snapshot)
                        log_step(f"Claude: {time.time() - t0:.1f}s")
                        show_decision_compact(decision)
                    except Exception as e:
                        log_step(f"[red]Analysis error: {e}[/red]")
                        time.sleep(interval)
                        continue

                    if decision.action.value == "SELL":
                        log_step("[bold red]Claude is BEARISH — skipping call entry.[/bold red]")
                        if not auto:
                            if not click.confirm("Claude is bearish. Buy calls anyway?"):
                                time.sleep(interval)
                                continue
                    elif decision.action.value == "HOLD":
                        log_step("[yellow]Claude is NEUTRAL — no clear bullish setup.[/yellow]")
                        if not auto:
                            if not click.confirm("Claude is neutral. Buy calls anyway?"):
                                time.sleep(interval)
                                continue
                    else:
                        log_step("[bold green]Claude is BULLISH — looking for contract.[/bold green]")

                    # Find the OTM call
                    current_price = float(df["close"].iloc[-1])
                    target_strike = current_price * 1.15
                    log_step(
                        f"Searching: strike ~${target_strike:.2f} (+15%), DTE 45–79..."
                    )
                    try:
                        opt = find_otm_call(config, symbol, current_price)
                    except Exception as e:
                        log_step(f"[red]Option chain error: {e}[/red]")
                        time.sleep(interval)
                        continue

                    if opt is None:
                        log_step(
                            f"[yellow]No contract found (strike ~${target_strike:.2f}, "
                            "DTE 45–79) — skipping.[/yellow]"
                        )
                        time.sleep(interval)
                        continue

                    # Check if we already hold this contract
                    already_held = any(
                        m.option_symbol == opt["symbol"] for m in managed
                    )
                    if already_held:
                        log_step(
                            f"[dim]Already holding {opt['symbol']} — not doubling up.[/dim]"
                        )
                        time.sleep(interval)
                        continue

                    actual_otm = opt["otm_pct"] * 100
                    console.print(
                        f"\n[bold]Contract:[/bold]  [cyan]{opt['symbol']}[/cyan]  "
                        f"strike ${opt['strike']:.2f} ([yellow]{actual_otm:+.1f}% OTM[/yellow])  "
                        f"exp {opt['expiry']} ({opt['dte']}d DTE)  "
                        f"{contracts} contract(s)"
                    )

                    # Confirm unless --auto
                    proceed = True
                    if not auto:
                        mode_label = (
                            "[yellow]PAPER[/yellow]" if config.is_paper else "[red]LIVE[/red]"
                        )
                        console.print(f"Mode: {mode_label}")
                        if not click.confirm(f"Buy {contracts}x {opt['symbol']}?"):
                            console.print("[yellow]Skipped.[/yellow]")
                            proceed = False

                    if proceed:
                        log_step(f"Submitting BUY {contracts}x {opt['symbol']}...")
                        try:
                            result = execute_option_buy(config, opt["symbol"], contracts)
                            console.print(
                                f"[green]Bought![/green]  ID: {result['order_id']}  "
                                f"({len(managed) + 1}/{max_positions} positions)"
                            )
                            log_step("Waiting 8s for fill...")
                            time.sleep(8)
                            managed.append(
                                _ManagedOption(
                                    option_symbol=opt["symbol"],
                                    contracts=contracts,
                                )
                            )
                        except Exception as e:
                            log_step(f"[red]Order error: {e}[/red]")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Options-swing monitoring stopped.[/yellow]")
        if managed:
            console.print(f"\n  [bold]{len(managed)} open position(s):[/bold]")
            for pos_obj in managed:
                try:
                    pos = get_position(config, pos_obj.option_symbol)
                    if pos:
                        pl = pos["unrealized_pl"]
                        plpc = pos["unrealized_plpc"]
                        pl_color = "green" if pl >= 0 else "red"
                        console.print(
                            f"  [cyan]{pos_obj.option_symbol}[/cyan]  "
                            f"[{pl_color}]${pl:+.2f} ({plpc:+.1%})[/{pl_color}]  "
                            f"peak ${pos_obj.peak_value:.2f}"
                        )
                    else:
                        console.print(f"  [dim]{pos_obj.option_symbol} — position not found[/dim]")
                except Exception:
                    console.print(f"  [dim]{pos_obj.option_symbol}[/dim]")
            console.print(
                "  [yellow]All positions remain OPEN. "
                "Re-run to resume monitoring or close via Alpaca dashboard.[/yellow]"
            )


@dataclass
class _Managed0DTE:
    """Tracks per-position state for 0DTE option positions."""
    option_symbol: str
    contracts: int
    entry_value: float   # market value right after fill (baseline for stop/target)
    peak_value: float = 0.0
    check_num: int = 0


@cli.command("0dte")
@click.argument("symbol")
@click.option("--paper/--live", default=True, help="Use paper or live trading mode.")
@click.option("--contracts", type=int, default=1,
              help="Contracts per entry (default 1, each = 100 shares).")
@click.option("--auto", is_flag=True, default=False,
              help="Auto-sell at profit target without prompting.")
@click.option("--interval", type=int, default=30,
              help="Seconds between position checks (default 30).")
@click.option("--refresh", type=int, default=120,
              help="Seconds between new-entry scans (default 120 = 2 min).")
@click.option("--max-positions", type=int, default=2,
              help="Max simultaneous open positions (default 2).")
@click.option("--stop-loss", "stop_loss", type=float, default=0.40,
              help="Exit if position value drops this fraction from entry (default 0.40 = 40%).")
@click.option("--profit-target", "profit_target", type=float, default=0.75,
              help="Exit if position value gains this fraction from entry (default 0.75 = 75%).")
@click.option("--close-before", "close_before", type=int, default=15,
              help="Force-close all positions N minutes before market close (default 15).")
@click.option("--bar-minutes", "bar_minutes",
              type=click.Choice(["1", "2", "3", "5"]), default="1",
              help="Bar size for signal scanning in minutes (default 1 — "
                   "minimum available from Alpaca REST API).")
def zero_dte(
    symbol: str,
    paper: bool,
    contracts: int,
    auto: bool,
    interval: int,
    refresh: int,
    max_positions: int,
    stop_loss: float,
    profit_target: float,
    close_before: int,
    bar_minutes: str,
):
    """Same-day (0DTE) options — buys calls/puts expiring today.

    \b
    ⚠  EXTREME RISK: 0DTE options can lose 100% of value by end of day.
    Time decay (theta) is severe — positions must be actively managed.

    \b
    Strategy:
      • Validates that 0DTE options exist for SYMBOL — exits immediately if not
      • Scans intraday 1-min bars for strong directional signals
      • Claude confirms each setup before any order is placed
      • Buys ATM calls (bullish signal) or puts (bearish signal) expiring today
      • Hard stop loss: exits if value drops --stop-loss % from entry
      • Profit target: exits (or prompts) at --profit-target % gain from entry
      • Force-closes ALL positions --close-before minutes before market close

    \b
    Only a handful of high-volume names have 0DTE liquidity: SPY, QQQ, SPX,
    AAPL, TSLA, NVDA, MSFT, AMZN. Most other stocks will error out.

    \b
    Examples:
      trading-bot 0dte SPY
      trading-bot 0dte AAPL --stop-loss 0.30 --profit-target 1.0 --auto
      trading-bot 0dte SPY --contracts 2 --max-positions 3 --close-before 20
    """
    symbol = symbol.upper()
    bar_min = int(bar_minutes)
    config = load_config()
    if not paper:
        config.trading_mode = "live"

    if not _check_market_open(config):
        return

    mode_tag = "[yellow]PAPER[/yellow]" if config.is_paper else "[red bold]LIVE[/red bold]"
    console.print(
        f"\n[bold red]0DTE OPTIONS[/bold red]  {mode_tag}  [cyan]{symbol}[/cyan]  "
        f"{contracts} contract(s)/entry  max {max_positions} open"
    )
    console.print(
        "[bold red]⚠  EXTREME RISK: Same-day expiry. "
        "Positions can lose 100% by close.[/bold red]"
    )
    console.print(
        f"[dim]Bars: {bar_min}min  "
        f"Stop loss: -{stop_loss:.0%}  "
        f"Profit target: +{profit_target:.0%}  "
        f"Force close: {close_before}min before market close  "
        f"Check every {interval}s  Scan every {refresh}s[/dim]\n"
    )

    # ── Validate 0DTE availability ────────────────────────────────────────────
    log_step(f"Checking 0DTE availability for [cyan]{symbol}[/cyan]...")
    try:
        available = check_0dte_available(config, symbol)
    except Exception as e:
        console.print(f"[red]Error checking option chain: {e}[/red]")
        return

    if not available:
        console.print(
            f"\n[bold red]ERROR: No 0DTE options found for {symbol}.[/bold red]\n"
            "\nSame-day expiring options only exist for a small set of high-volume\n"
            "stocks and ETFs that have daily (Mon–Fri) expiry cycles, such as:\n"
            "  SPY  QQQ  AAPL  TSLA  NVDA  MSFT  AMZN  GOOGL  META  AMD\n"
            "\nMost mid/small-cap stocks only have weekly or monthly expiries.\n"
            "[yellow]Exiting.[/yellow]"
        )
        return

    log_step(f"[green]0DTE options confirmed for {symbol}.[/green]")

    # ── Get market close time ─────────────────────────────────────────────────
    market_close = None
    try:
        clock = get_market_clock(config)
        market_close = clock["next_close"]
        close_str = market_close.strftime("%I:%M %p ET")
        force_str = (market_close - timedelta(minutes=close_before)).strftime("%I:%M %p ET")
        log_step(f"Market closes {close_str}  —  force-close at {force_str}")
    except Exception as e:
        log_step(f"[yellow]Could not fetch close time: {e} — time stop disabled.[/yellow]")

    managed: list[_Managed0DTE] = []
    last_scan = 0.0   # force immediate first scan

    try:
        while True:
            now_ts = time.time()
            do_scan = (now_ts - last_scan) >= refresh

            # ── TIME-BASED FORCE CLOSE ────────────────────────────────────────
            if market_close is not None:
                now_aware = datetime.now(timezone.utc).astimezone(market_close.tzinfo)
                mins_left = (market_close - now_aware).total_seconds() / 60

                if mins_left <= close_before:
                    if managed:
                        console.print(
                            f"\n[bold red]FORCE CLOSE: {mins_left:.0f}min to market close — "
                            f"closing all {len(managed)} position(s).[/bold red]"
                        )
                        for pos_obj in list(managed):
                            try:
                                pos = get_position(config, pos_obj.option_symbol)
                                pl_str = (
                                    f"${pos['unrealized_pl']:+.2f} ({pos['unrealized_plpc']:+.1%})"
                                    if pos else "unknown P&L"
                                )
                                close_option_position(config, pos_obj.option_symbol)
                                console.print(
                                    f"  [yellow]Closed {pos_obj.option_symbol}  {pl_str}[/yellow]"
                                )
                            except Exception as e:
                                console.print(
                                    f"  [red]Close error {pos_obj.option_symbol}: {e}[/red]"
                                )
                        managed.clear()
                    else:
                        console.print(
                            f"[dim]{mins_left:.0f}min to market close — "
                            "no open positions. Session complete.[/dim]"
                        )
                    console.print("[bold]0DTE session ended.[/bold]")
                    return

            # ── POSITION MONITORING ───────────────────────────────────────────
            closed_symbols = []
            for pos_obj in list(managed):
                pos_obj.check_num += 1
                option_symbol = pos_obj.option_symbol

                try:
                    position = get_position(config, option_symbol)
                except Exception as e:
                    log_step(f"[red]Fetch error ({option_symbol}): {e}[/red]")
                    continue

                if position is None:
                    console.print(
                        f"[yellow]{option_symbol} — gone (expired or closed externally).[/yellow]"
                    )
                    closed_symbols.append(option_symbol)
                    continue

                mkt_val = position["market_value"]
                plpc = position["unrealized_plpc"]
                pl_dollar = position["unrealized_pl"]

                if mkt_val > pos_obj.peak_value:
                    pos_obj.peak_value = mkt_val

                pl_color = "green" if pl_dollar >= 0 else "red"
                log_step(
                    f"#{pos_obj.check_num} [cyan]{option_symbol}[/cyan]  "
                    f"val ${mkt_val:.2f}  "
                    f"[{pl_color}]{plpc:+.1%} (${pl_dollar:+.2f})[/{pl_color}]  "
                    f"entry ${pos_obj.entry_value:.2f}  "
                    f"peak ${pos_obj.peak_value:.2f}"
                )

                # Stop loss: value dropped --stop-loss % from entry
                if pos_obj.entry_value > 0:
                    change_from_entry = (mkt_val - pos_obj.entry_value) / pos_obj.entry_value
                    if change_from_entry <= -stop_loss:
                        log_step(
                            f"[bold red]STOP LOSS {option_symbol}  "
                            f"{change_from_entry:.1%} from entry  "
                            f"(${mkt_val:.2f} vs entry ${pos_obj.entry_value:.2f})[/bold red]"
                        )
                        try:
                            close_option_position(config, option_symbol)
                            console.print(
                                f"[red]{option_symbol} stopped out.  "
                                f"P&L: ${pl_dollar:+.2f} ({plpc:+.1%})[/red]"
                            )
                        except Exception as e:
                            console.print(f"[red]Close error: {e}[/red]")
                        closed_symbols.append(option_symbol)
                        continue

                    # Profit target: value up --profit-target % from entry
                    if change_from_entry >= profit_target:
                        console.print(
                            f"\n[bold green]{option_symbol} hit profit target "
                            f"+{change_from_entry:.0%}![/bold green]  (${pl_dollar:+.2f})"
                        )
                        sell = auto or click.confirm("Sell now?")
                        if sell:
                            try:
                                close_option_position(config, option_symbol)
                                console.print(
                                    f"[bold green]Sold {option_symbol}!  "
                                    f"P&L: ${pl_dollar:+.2f} ({plpc:+.1%})[/bold green]"
                                )
                            except Exception as e:
                                console.print(f"[red]Close error: {e}[/red]")
                            closed_symbols.append(option_symbol)
                            continue
                        else:
                            console.print(
                                f"[dim]Holding. Hard stop still active "
                                f"at -{stop_loss:.0%} from entry.[/dim]"
                            )

            if closed_symbols:
                managed = [m for m in managed if m.option_symbol not in closed_symbols]
                console.print(f"[dim]Active: {len(managed)}/{max_positions}[/dim]")

            # ── ENTRY SCAN ────────────────────────────────────────────────────
            if do_scan:
                last_scan = now_ts
                open_count = len(managed)

                if open_count >= max_positions:
                    log_step(
                        f"[dim]Position limit ({open_count}/{max_positions}) — "
                        "skipping scan.[/dim]"
                    )
                else:
                    log_step(
                        f"[bold]0DTE scan[/bold]  ({open_count}/{max_positions} open)  "
                        f"Fetching {bar_min}-min bars for [cyan]{symbol}[/cyan]..."
                    )

                    try:
                        df = fetch_intraday_bars(config, symbol, bar_minutes=bar_min)
                    except Exception as e:
                        log_step(f"[red]Data error: {e}[/red]")
                        time.sleep(interval)
                        continue

                    if len(df) < 15:
                        log_step(f"[yellow]Only {len(df)} bars — need 15+.[/yellow]")
                        time.sleep(interval)
                        continue

                    try:
                        snapshot = compute_indicators(df, symbol, fast=True)
                        show_indicators_compact(snapshot)
                    except Exception as e:
                        log_step(f"[red]Indicator error: {e}[/red]")
                        time.sleep(interval)
                        continue

                    signal = scan_entry(snapshot)
                    if signal.direction == "none":
                        log_step(
                            f"[dim]No signal ({signal.strength}/5). "
                            f"{', '.join(signal.reasons)}[/dim]"
                        )
                        time.sleep(interval)
                        continue

                    sig_color = "green" if signal.direction == "long" else "red"
                    log_step(
                        f"[{sig_color} bold]SIGNAL: {signal.direction.upper()} "
                        f"({signal.strength}/5)[/{sig_color} bold]  "
                        + "  ".join(signal.reasons)
                    )

                    log_step("Asking Claude to confirm 0DTE setup...")
                    try:
                        t0 = time.time()
                        decision = get_scalp_analysis(
                            config, snapshot,
                            signal_direction=signal.direction,
                            signal_strength=signal.strength,
                            signal_reasons=signal.reasons,
                        )
                        log_step(f"Claude: {time.time() - t0:.1f}s")
                        show_decision_compact(decision)
                    except Exception as e:
                        log_step(f"[red]Claude error: {e}[/red]")
                        time.sleep(interval)
                        continue

                    if decision.action.value == "HOLD":
                        log_step("[yellow]Claude rejected — skipping.[/yellow]")
                        time.sleep(interval)
                        continue

                    direction = "long" if decision.action.value == "BUY" else "short"
                    current_price = snapshot.current_price
                    opt_type = "CALL" if direction == "long" else "PUT"

                    log_step(
                        f"[magenta]Looking for 0DTE {opt_type} "
                        f"on {symbol} @ ${current_price:.2f}...[/magenta]"
                    )
                    try:
                        opt = find_best_option(
                            config, symbol, direction, current_price,
                            min_dte=0, max_dte=0,
                        )
                    except Exception as e:
                        log_step(f"[red]Option lookup error: {e}[/red]")
                        time.sleep(interval)
                        continue

                    if opt is None:
                        log_step(
                            f"[yellow]No 0DTE {opt_type} found within ±3% of "
                            f"${current_price:.2f}. Strike unavailable today.[/yellow]"
                        )
                        time.sleep(interval)
                        continue

                    if any(m.option_symbol == opt["symbol"] for m in managed):
                        log_step(f"[dim]Already holding {opt['symbol']} — skipping.[/dim]")
                        time.sleep(interval)
                        continue

                    opt_color = "green" if direction == "long" else "red"
                    console.print(
                        f"\n[bold]0DTE Contract:[/bold]  "
                        f"[{opt_color}][cyan]{opt['symbol']}[/cyan]  "
                        f"{opt_type} strike ${opt['strike']:.2f}  "
                        f"exp {opt['expiry']}[/{opt_color}]  "
                        f"{contracts} contract(s)"
                    )

                    proceed = True
                    if not auto:
                        mode_label = (
                            "[yellow]PAPER[/yellow]" if config.is_paper else "[red]LIVE[/red]"
                        )
                        console.print(f"Mode: {mode_label}")
                        if not click.confirm(f"Buy {contracts}x {opt['symbol']}?"):
                            console.print("[yellow]Skipped.[/yellow]")
                            proceed = False

                    if proceed:
                        log_step(f"Submitting BUY {contracts}x {opt['symbol']}...")
                        try:
                            result = execute_option_buy(config, opt["symbol"], contracts)
                            console.print(
                                f"[green]Bought![/green]  ID: {result['order_id']}  "
                                f"({len(managed) + 1}/{max_positions} positions)"
                            )
                            log_step("Waiting 5s for fill...")
                            time.sleep(5)
                            pos = get_position(config, opt["symbol"])
                            entry_val = pos["market_value"] if pos else 0.0
                            managed.append(
                                _Managed0DTE(
                                    option_symbol=opt["symbol"],
                                    contracts=contracts,
                                    entry_value=entry_val,
                                    peak_value=entry_val,
                                )
                            )
                        except Exception as e:
                            log_step(f"[red]Order error: {e}[/red]")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]0DTE session stopped.[/yellow]")
        if managed:
            console.print(
                f"\n  [bold red]⚠  {len(managed)} open 0DTE position(s) — "
                "EXPIRE TODAY:[/bold red]"
            )
            for pos_obj in managed:
                try:
                    pos = get_position(config, pos_obj.option_symbol)
                    if pos:
                        pl = pos["unrealized_pl"]
                        plpc = pos["unrealized_plpc"]
                        pl_color = "green" if pl >= 0 else "red"
                        console.print(
                            f"  [cyan]{pos_obj.option_symbol}[/cyan]  "
                            f"[{pl_color}]${pl:+.2f} ({plpc:+.1%})[/{pl_color}]"
                        )
                    else:
                        console.print(f"  [dim]{pos_obj.option_symbol} — not found[/dim]")
                except Exception:
                    console.print(f"  [dim]{pos_obj.option_symbol}[/dim]")
            console.print(
                "  [bold red]Close these NOW via Alpaca or they expire worthless at EOD![/bold red]"
            )
