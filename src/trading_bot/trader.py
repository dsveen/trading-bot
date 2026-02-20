from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetOrdersRequest

from trading_bot.config import Config
from trading_bot.models import Action, TradeDecision


def execute_bracket_order(
    config: Config, symbol: str, decision: TradeDecision
) -> dict:
    """Submit a bracket limit order (entry + stop-loss + take-profit)."""
    if decision.action == Action.HOLD:
        return {"status": "skipped", "reason": "HOLD decision"}

    client = TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.is_paper,
    )

    side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL

    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=decision.suggested_shares,
        side=side,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=round(decision.suggested_entry_price, 2),
        order_class=OrderClass.BRACKET,
        stop_loss={"stop_price": round(decision.stop_loss, 2)},
        take_profit={"limit_price": round(decision.take_profit, 2)},
    )

    order = client.submit_order(order_request)

    return {
        "status": "submitted",
        "order_id": str(order.id),
        "symbol": symbol,
        "side": side.value,
        "qty": decision.suggested_shares,
        "limit_price": decision.suggested_entry_price,
        "stop_loss": decision.stop_loss,
        "take_profit": decision.take_profit,
        "order_class": "bracket",
    }


def execute_market_order(
    config: Config, symbol: str, decision: TradeDecision
) -> dict:
    """Submit a market order for instant fill (scalping)."""
    if decision.action == Action.HOLD:
        return {"status": "skipped", "reason": "HOLD decision"}

    client = TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.is_paper,
    )

    side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=decision.suggested_shares,
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    order = client.submit_order(order_request)

    return {
        "status": "filled",
        "order_id": str(order.id),
        "symbol": symbol,
        "side": side.value,
        "qty": decision.suggested_shares,
    }


def _cancel_open_orders(client: TradingClient, symbol: str) -> int:
    """Cancel all open orders for a symbol. Returns the number cancelled."""
    try:
        open_orders = client.get_orders(
            GetOrdersRequest(symbol=symbol, status=QueryOrderStatus.OPEN)
        )
        for order in open_orders:
            try:
                client.cancel_order_by_id(order.id)
            except Exception:
                pass
        return len(open_orders)
    except Exception:
        return 0


def close_position(config: Config, symbol: str) -> dict:
    """Cancel any open orders for the symbol, then close the position."""
    client = TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.is_paper,
    )

    _cancel_open_orders(client, symbol)
    order = client.close_position(symbol)

    return {
        "status": "closing",
        "order_id": str(order.id) if hasattr(order, "id") else "unknown",
        "symbol": symbol,
    }


def execute_option_buy(
    config: Config,
    option_symbol: str,
    contracts: int = 1,
) -> dict:
    """Buy option contracts (buy-to-open) via market order.

    Each contract controls 100 shares. Use 1-2 contracts for small accounts.
    option_symbol: OCC-formatted symbol, e.g. 'AAPL241220C00170000'
    """
    client = TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.is_paper,
    )

    order_request = MarketOrderRequest(
        symbol=option_symbol,
        qty=contracts,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )

    order = client.submit_order(order_request)

    return {
        "status": "submitted",
        "order_id": str(order.id),
        "symbol": option_symbol,
        "contracts": contracts,
    }


def close_option_position(config: Config, option_symbol: str) -> dict:
    """Cancel any open orders for the contract, then sell-to-close."""
    client = TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=config.is_paper,
    )

    _cancel_open_orders(client, option_symbol)
    order = client.close_position(option_symbol)

    return {
        "status": "closing",
        "order_id": str(order.id) if hasattr(order, "id") else "unknown",
        "symbol": option_symbol,
    }
