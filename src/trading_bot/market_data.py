from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, QueryOrderStatus
from alpaca.trading.requests import GetOptionContractsRequest, GetOrdersRequest

from trading_bot.config import Config


def get_data_client(config: Config) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )


def get_trading_client(config: Config) -> TradingClient:
    return TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.is_paper,
    )


def fetch_bars(config: Config, symbol: str, days: int = 100) -> pd.DataFrame:
    """Fetch daily bars (for swing analysis)."""
    client = get_data_client(config)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days),
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    return df


def fetch_intraday_bars(
    config: Config,
    symbol: str,
    bar_minutes: int = 1,
    lookback_minutes: int = 90,
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars at a configurable minute resolution.

    bar_minutes: bar width in minutes (1 is the minimum the Alpaca REST API supports).
    lookback_minutes: how far back to fetch — auto-set to at least 60 * bar_minutes
    so there are always enough bars for indicators.
    """
    # Always keep enough bars for the slowest indicator (50-period needs 50 bars minimum)
    effective_lookback = max(lookback_minutes, bar_minutes * 60)
    client = get_data_client(config)
    timeframe = TimeFrame(bar_minutes, TimeFrameUnit.Minute)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=datetime.now() - timedelta(minutes=effective_lookback),
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    return df


def fetch_minute_bars(config: Config, symbol: str, minutes: int = 120) -> pd.DataFrame:
    """Fetch 1-minute bars for intraday/scalping."""
    client = get_data_client(config)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime.now() - timedelta(minutes=minutes),
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    return df


def fetch_latest_price(config: Config, symbol: str) -> float:
    client = get_data_client(config)
    request = StockLatestBarRequest(symbol_or_symbols=symbol)
    bar = client.get_stock_latest_bar(request)
    return float(bar[symbol].close)


def fetch_account(config: Config) -> dict:
    client = get_trading_client(config)
    account = client.get_account()
    return {
        "equity": float(account.equity),
        "buying_power": float(account.buying_power),
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "day_trade_count": account.daytrade_count,
    }


def fetch_positions(config: Config) -> list[dict]:
    client = get_trading_client(config)
    positions = client.get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
            "side": p.side.value,
        }
        for p in positions
    ]


def get_market_clock(config: Config) -> dict:
    """Return current market status from Alpaca's clock endpoint."""
    client = get_trading_client(config)
    clock = client.get_clock()
    return {
        "is_open": clock.is_open,
        "next_open": clock.next_open,    # timezone-aware datetime
        "next_close": clock.next_close,  # timezone-aware datetime
        "timestamp": clock.timestamp,    # current server time
    }


def get_position(config: Config, symbol: str) -> dict | None:
    """Get current position for a specific symbol, or None."""
    for p in fetch_positions(config):
        if p["symbol"] == symbol:
            return p
    return None


def fetch_option_contracts(
    config: Config,
    symbol: str,
    direction: str,
    current_price: float,
    min_dte: int = 2,
    max_dte: int = 14,
) -> list:
    """Fetch near-ATM option contracts for given direction.

    direction: "long" → CALL, "short" → PUT
    Looks for contracts within ±3% of current price expiring in min_dte–max_dte days.
    """
    client = get_trading_client(config)
    contract_type = ContractType.CALL if direction == "long" else ContractType.PUT
    today = date.today()

    request = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        expiration_date_gte=today + timedelta(days=min_dte),
        expiration_date_lte=today + timedelta(days=max_dte),
        type=contract_type,
        strike_price_gte=str(round(current_price * 0.97, 2)),
        strike_price_lte=str(round(current_price * 1.03, 2)),
        limit=20,
    )

    result = client.get_option_contracts(request)
    return result.option_contracts if result.option_contracts else []


def find_otm_call(
    config: Config,
    symbol: str,
    current_price: float,
    otm_pct: float = 0.15,
    min_dte: int = 45,
    max_dte: int = 79,
) -> dict | None:
    """Find a CALL option approximately otm_pct OTM expiring in min_dte–max_dte days.

    Prefers the 45–60 DTE window. Returns the contract with strike closest to
    current_price * (1 + otm_pct) within the DTE range, or None if not found.
    """
    client = get_trading_client(config)
    target_strike = current_price * (1 + otm_pct)
    today = date.today()

    # Search ±8% around target strike to account for available strike intervals
    request = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        expiration_date_gte=today + timedelta(days=min_dte),
        expiration_date_lte=today + timedelta(days=max_dte),
        type=ContractType.CALL,
        strike_price_gte=str(round(current_price * (1 + otm_pct - 0.08), 2)),
        strike_price_lte=str(round(current_price * (1 + otm_pct + 0.08), 2)),
        limit=50,
    )

    result = client.get_option_contracts(request)
    contracts = result.option_contracts if result.option_contracts else []

    if not contracts:
        return None

    def score(c):
        strike = float(c.strike_price)
        expiry = c.expiration_date
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
        dte = (expiry - today).days
        strike_dist = abs(strike - target_strike)
        # Prefer 45–60 DTE range over 61–79
        dte_penalty = 0.0 if dte <= 60 else (dte - 60) * 0.5
        return (strike_dist + dte_penalty, dte)

    contracts.sort(key=score)
    best = contracts[0]

    expiry = best.expiration_date
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry[:10], "%Y-%m-%d").date()

    strike = float(best.strike_price)
    return {
        "symbol": best.symbol,
        "strike": strike,
        "expiry": str(expiry),
        "dte": (expiry - today).days,
        "otm_pct": (strike - current_price) / current_price,
    }


def check_0dte_available(config: Config, symbol: str) -> bool:
    """Return True if any same-day (0DTE) call option exists for this symbol.

    Most stocks only have weekly or monthly expiries. Only high-volume names
    like SPY, QQQ, AAPL, TSLA, NVDA, MSFT offer daily expiries.
    """
    client = get_trading_client(config)
    today = date.today()
    try:
        request = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date_gte=today,
            expiration_date_lte=today,
            type=ContractType.CALL,
            limit=1,
        )
        result = client.get_option_contracts(request)
        return bool(result.option_contracts)
    except Exception:
        return False


def find_best_option(
    config: Config,
    symbol: str,
    direction: str,
    current_price: float,
    min_dte: int = 2,
    max_dte: int = 14,
) -> dict | None:
    """Find the nearest-expiry, closest-to-ATM option contract.

    Returns a dict with symbol, strike, expiry, type, dte — or None if no contract found.
    """
    contracts = fetch_option_contracts(config, symbol, direction, current_price, min_dte, max_dte)
    if not contracts:
        return None

    today = date.today()

    def score(c):
        strike = float(c.strike_price)
        expiry = c.expiration_date
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
        days_out = (expiry - today).days
        atm_dist = abs(strike - current_price)
        return (days_out, atm_dist)

    contracts.sort(key=score)
    best = contracts[0]

    expiry = best.expiration_date
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry[:10], "%Y-%m-%d").date()

    return {
        "symbol": best.symbol,
        "strike": float(best.strike_price),
        "expiry": str(expiry),
        "type": "CALL" if direction == "long" else "PUT",
        "dte": (expiry - today).days,
    }


def fetch_trade_history(config: Config, days: int = 30) -> list[dict]:
    """Fetch filled orders from the last N days and compute realized P&L per symbol.

    Returns a list of dicts, one per symbol that had any filled order, with:
      symbol, buy_qty, avg_buy_price, sell_qty, avg_sell_price,
      matched_qty, realized_pl, realized_plpc, open_qty
    """
    client = get_trading_client(config)
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)

    orders = client.get_orders(
        GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=since,
            limit=500,
        )
    )

    # Group fills by symbol → side
    buys: dict[str, list[tuple[float, float]]] = defaultdict(list)   # symbol → [(qty, price)]
    sells: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for order in orders:
        if order.status.value not in ("filled", "partially_filled"):
            continue
        if not order.filled_qty or not order.filled_avg_price:
            continue

        qty = float(order.filled_qty)
        price = float(order.filled_avg_price)
        symbol = order.symbol

        if order.side.value == "buy":
            buys[symbol].append((qty, price))
        else:
            sells[symbol].append((qty, price))

    all_symbols = sorted(set(list(buys.keys()) + list(sells.keys())))
    results = []

    for symbol in all_symbols:
        buy_fills = buys[symbol]
        sell_fills = sells[symbol]

        total_buy_qty = sum(q for q, _ in buy_fills)
        total_sell_qty = sum(q for q, _ in sell_fills)

        avg_buy = (
            sum(q * p for q, p in buy_fills) / total_buy_qty
            if total_buy_qty > 0 else 0.0
        )
        avg_sell = (
            sum(q * p for q, p in sell_fills) / total_sell_qty
            if total_sell_qty > 0 else 0.0
        )

        matched_qty = min(total_buy_qty, total_sell_qty)
        realized_pl = matched_qty * (avg_sell - avg_buy) if matched_qty > 0 else 0.0
        realized_plpc = (
            (avg_sell - avg_buy) / avg_buy if avg_buy > 0 and matched_qty > 0 else 0.0
        )

        results.append({
            "symbol": symbol,
            "buy_qty": total_buy_qty,
            "avg_buy_price": avg_buy,
            "sell_qty": total_sell_qty,
            "avg_sell_price": avg_sell,
            "matched_qty": matched_qty,
            "realized_pl": realized_pl,
            "realized_plpc": realized_plpc,
            "open_qty": total_buy_qty - total_sell_qty,
        })

    return results
