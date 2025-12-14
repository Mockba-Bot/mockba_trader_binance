import os
import math
import json
import threading
import time
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException
import redis
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot.send_bot_message import send_bot_message
from logs.log_config import binance_trader_logger as logger

load_dotenv()

# Initialize Binance Futures client
client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_SECRET_KEY"),
    testnet=False
)

# Initialize Redis connection
redis_url = os.getenv("REDIS_URL")
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except redis.ConnectionError as e:
        logger.warning(f"Redis not available (optional caching disabled): {e}")
        redis_client = None
else:
    logger.info("Redis not configured (optional caching disabled)")
    redis_client = None

# Risk parameters
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.5"))

# In-memory active trades and threads
_active_trades = {}
_monitoring_threads = {}  # {symbol: {'thread': Thread, 'stop_event': Event}}
_lock = threading.Lock()

def get_confidence_level(confidence: float) -> str:
    if confidence >= 3.0:
        return "🚀 VERY STRONG"
    elif confidence >= 2.0:
        return "💪 STRONG"
    elif confidence >= 1.8:
        return "👍 MODERATE"
    else:
        return "⚠️ WEAK"

def get_futures_exchange_info(symbol: str):
    cache_key = f"futures_info_{symbol}"
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                filters = {f['filterType']: f for f in s['filters']}
                result = {
                    'pricePrecision': s['pricePrecision'],
                    'quantityPrecision': s['quantityPrecision'],
                    'stepSize': float(filters['LOT_SIZE']['stepSize']),
                    'minQty': float(filters['LOT_SIZE']['minQty']),
                    'maxQty': float(filters['LOT_SIZE']['maxQty']),
                    'minNotional': float(filters.get('MIN_NOTIONAL', {}).get('notional', 0)) or 5.0
                }
                if redis_client:
                    redis_client.setex(cache_key, 3600, json.dumps(result))
                return result
    except Exception as e:
        logger.error(f"Failed to get futures info for {symbol}: {e}")
    return None

def get_available_balance(asset: str = "USDT") -> float:
    try:
        account = client.futures_account()
        if 'assets' not in account:
            logger.error("Invalid futures account response")
            return 0.0
        for asset_info in account['assets']:
            if asset_info['asset'] == asset:
                balance = float(asset_info.get('availableBalance', 0))
                logger.info(f"Available {asset} balance: {balance}")
                return balance
        logger.warning(f"Asset {asset} not found in futures account")
    except Exception as e:
        logger.error(f"Failed to get balance: {e}")
    return 0.0

def set_leverage(symbol: str, leverage: int):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logger.info(f"Set leverage for {symbol} to {leverage}x")
    except Exception as e:
        logger.error(f"Failed to set leverage for {symbol}: {e}")

def round_step_size(quantity: float, step_size: float) -> float:
    if step_size <= 0:
        return quantity
    precision = max(0, int(round(-math.log(step_size, 10), 0)))
    return round(quantity - (quantity % step_size), precision)

def calculate_position_size_with_margin_cap(signal: dict, available_balance: float, leverage: int, symbol_info: dict) -> float:
    entry = float(signal['entry'])
    sl = float(signal['stop_loss'])
    side = signal['side'].upper()

    risk_amount = available_balance * (RISK_PER_TRADE_PCT / 100)
    risk_per_unit = abs(entry - sl)

    if risk_per_unit <= 0:
        logger.warning("Invalid stop loss placement")
        return 0.0
    if risk_per_unit < 1e-10:
        logger.warning("Risk per unit too small, skipping trade")
        return 0.0

    qty_by_risk = risk_amount / risk_per_unit
    max_notional = available_balance * leverage * 0.99
    if entry <= 0:
        logger.warning("Invalid entry price")
        return 0.0
    qty_by_margin = max_notional / entry

    qty = min(qty_by_risk, qty_by_margin)
    qty = round_step_size(qty, symbol_info['stepSize'])
    if qty < symbol_info['minQty']:
        logger.warning(f"Qty {qty} below minQty {symbol_info['minQty']}")
        return 0.0

    notional = qty * entry
    if notional < symbol_info['minNotional']:
        logger.warning(f"Notional ${notional:.2f} below min ${symbol_info['minNotional']}")
        return 0.0

    return qty

def close_position_market(symbol: str, side: str, qty: float, info: dict):
    qty = round_step_size(qty, info['stepSize'])
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=qty,
            positionSide='BOTH'
        )
        logger.info(f"CloseOperation filled: {order['orderId']} for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Failed to close position for {symbol}: {e}")
        return False

def get_redis_trade_key(symbol: str) -> str:
    return f"trade_monitor:{symbol}"

def store_trade_in_redis(symbol: str, trade_data: dict):
    """Store trade metadata in Redis with 24h TTL"""
    if redis_client:
        safe_data = {
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'qty': trade_data['qty'],
            'tp_price': trade_data['tp_price'],
            'sl_price': trade_data['sl_price'],
            'price_precision': trade_data['symbol_info']['pricePrecision'],
            'step_size': trade_data['symbol_info']['stepSize'],
            'min_notional': trade_data['symbol_info']['minNotional']
        }
        redis_client.setex(get_redis_trade_key(symbol), 86400, json.dumps(safe_data))

def load_trade_from_redis(symbol: str) -> dict | None:
    """Load trade data from Redis (for recovery)"""
    if not redis_client:
        return None
    data = redis_client.get(get_redis_trade_key(symbol))
    if data:
        loaded = json.loads(data)
        loaded['symbol_info'] = {
            'pricePrecision': loaded.pop('price_precision'),
            'stepSize': loaded.pop('step_size'),
            'minNotional': loaded.pop('min_notional'),
            'quantityPrecision': 0,
            'minQty': 0.0
        }
        return loaded
    return None

def remove_trade_from_redis(symbol: str):
    """Remove trade from Redis"""
    if redis_client:
        redis_client.delete(get_redis_trade_key(symbol))

def monitor_trade(symbol: str, trade_data: dict, stop_event: threading.Event):
    side = trade_data['side']
    qty = trade_data['qty']
    tp_price = trade_data['tp_price']
    sl_price = trade_data['sl_price']
    info = trade_data['symbol_info']

    logger.info(f"Started monitor for {symbol} | TP: {tp_price}, SL: {sl_price}")

    while not stop_event.is_set():
        try:
            mark_data = client.futures_mark_price(symbol=symbol)
            current_price = float(mark_data['markPrice'])

            should_exit = False
            reason = ""

            if side == "BUY":
                if current_price >= tp_price:
                    should_exit = True
                    reason = "TP"
                elif current_price <= sl_price:
                    should_exit = True
                    reason = "SL"
            else:  # SELL
                if current_price <= tp_price:
                    should_exit = True
                    reason = "TP"
                elif current_price >= sl_price:
                    should_exit = True
                    reason = "SL"

            if should_exit:
                close_side = SIDE_SELL if side == "BUY" else SIDE_BUY
                success = close_position_market(symbol, close_side, qty, info)
                if success:
                    logger.info(f"✅ {reason} hit for {symbol} at {current_price:.4f}")
                    msg = (
                        f"🚨 BINANCE - Trade Closed!\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {side}\n"
                        f"Close Reason: {reason}\n"
                        f"Exit Price: {current_price:.4f}\n"
                        f"Qty: {qty:.6f}"
                    )
                    send_bot_message(int(os.getenv("TELEGRAM_CHAT_ID")), msg)
                else:
                    logger.error(f"Failed to close {symbol} on {reason}")
                break

            for _ in range(15):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in monitor for {symbol}: {e}")
            if not stop_event.is_set():
                time.sleep(5)

    with _lock:
        _active_trades.pop(symbol, None)
        _monitoring_threads.pop(symbol, None)
    remove_trade_from_redis(symbol)
    logger.info(f"Monitor cleanly stopped for {symbol}")

def place_futures_order(signal: dict):
    print(signal)
    symbol = signal['symbol']
    side = signal['side'].upper()
    
    leverage = signal.get('leverage')
    if leverage is None or leverage <= 0:
        logger.error(f"Invalid leverage in signal: {leverage}")
        return None
   
    info = get_futures_exchange_info(symbol)
    if not info:
        logger.error(f"Could not get symbol info for {symbol}")
        return None

    available_balance = get_available_balance()
    if available_balance <= 15.0:
        logger.error(f"Insufficient balance: ${available_balance:.2f}")
        return None

    qty = calculate_position_size_with_margin_cap(signal, available_balance, leverage, info)
    if qty <= 0:
        logger.warning(f"Position size calculation failed for {symbol}")
        return None

    entry_price = float(signal['entry'])
    notional = qty * entry_price
    max_safe_notional = available_balance * leverage * 0.95
    if notional < 10 or notional > max_safe_notional:
        logger.warning(f"Notional ${notional:.2f} outside safe range (max: ${max_safe_notional:.2f})")
        return None

    set_leverage(symbol, leverage)

    order_side = SIDE_BUY if side == 'BUY' else SIDE_SELL

    entry_price = round(entry_price, info['pricePrecision'])
    tp_price = round(float(signal['take_profit']), info['pricePrecision'])
    sl_price = round(float(signal['stop_loss']), info['pricePrecision'])

    if side == 'BUY':
        if tp_price <= entry_price or sl_price >= entry_price:
            logger.warning(f"Invalid TP/SL for BUY: TP={tp_price}, Entry={entry_price}, SL={sl_price}")
            return None
    else:
        if tp_price >= entry_price or sl_price <= entry_price:
            logger.warning(f"Invalid TP/SL for SELL: TP={tp_price}, Entry={entry_price}, SL={sl_price}")
            return None

    try:
        pos_info = client.futures_position_information(symbol=symbol)
        for pos in pos_info:
            if float(pos['positionAmt']) != 0:
                logger.warning(f"Position already exists for {symbol}, skipping")
                return None

        logger.info(f"Placing MARKET {side} for {symbol} | Qty: {qty} | Leverage: {leverage}x | Notional: ${notional:.2f}")
        entry_order = client.futures_create_order(
            symbol=symbol,
            type=ORDER_TYPE_MARKET,
            side=order_side,
            quantity=qty,
            positionSide='BOTH'
        )
        entry_id = entry_order['orderId']
        filled_price = float(entry_order.get('avgPrice') or entry_order.get('price') or entry_price)
        logger.info(f"Entry filled: {entry_id} @ {filled_price}")

        time.sleep(3)

        pos_info = client.futures_position_information(symbol=symbol)
        current_amt = float(pos_info[0]['positionAmt'])
        if abs(current_amt) < qty * 0.9:
            logger.error(f"Position size mismatch. Expected ~{qty}, got {current_amt}")
            return None

        trade_data = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "symbol_info": info
        }

        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_trade,
            args=(symbol, trade_data, stop_event),
            daemon=True
        )
        monitor_thread.start()

        with _lock:
            _active_trades[symbol] = trade_data
            _monitoring_threads[symbol] = {
                'thread': monitor_thread,
                'stop_event': stop_event
            }

        store_trade_in_redis(symbol, trade_data)

        confirmation_msg = (
            f"🚨 BINANCE - Signal Detected!\n"
            f"✅ POSITION OPENED\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Qty: {qty:.6f}\n"
            f"Entry: {filled_price:.4f}\n"
            f"TP: {tp_price:.4f}\n"
            f"SL: {sl_price:.4f}\n"
            f"Notional: ${notional:.2f}\n"
            f"Leverage: {leverage}x\n"
            f"Risk: {RISK_PER_TRADE_PCT}% of balance\n"
            f"ℹ️ Auto-close on TP/SL via price monitor"
        )
        send_bot_message(int(os.getenv("TELEGRAM_CHAT_ID")), confirmation_msg)

        return entry_id

    except BinanceAPIException as e:
        logger.error(f"Binance API error for {symbol}: {e.message} (code {e.code})")
        if e.code == -2019:
            logger.error("Insufficient margin")
        elif e.code == -1111:
            logger.error("Precision/step size error")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing orders for {symbol}: {e}")
        return None

def start_external_close_watcher(interval_seconds=15):
    def _watch():
        while True:
            try:
                with _lock:
                    symbols = list(_monitoring_threads.keys())

                for symbol in symbols:
                    try:
                        pos_info = client.futures_position_information(symbol=symbol)
                        position_amt = float(pos_info[0]['positionAmt'])

                        if position_amt == 0:
                            logger.info(f"Position for {symbol} closed externally. Stopping monitor.")
                            with _lock:
                                thread_info = _monitoring_threads.get(symbol)
                                if thread_info:
                                    thread_info['stop_event'].set()
                    except Exception as e:
                        logger.debug(f"External close check error for {symbol}: {e}")

                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"External close watcher error: {e}")
                time.sleep(interval_seconds)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    logger.info(f"External close watcher started (every {interval_seconds}s)")

def recover_trades_on_startup():
    if not redis_client:
        return
    try:
        keys = redis_client.keys("trade_monitor:*")
        for key in keys:
            symbol_bytes = key.split(b":")[-1]
            symbol = symbol_bytes.decode()
            trade_data = load_trade_from_redis(symbol)
            if trade_data:
                try:
                    pos_info = client.futures_position_information(symbol=symbol)
                    amt = float(pos_info[0]['positionAmt'])
                    if amt != 0:
                        stop_event = threading.Event()
                        monitor_thread = threading.Thread(
                            target=monitor_trade,
                            args=(symbol, trade_data, stop_event),
                            daemon=True
                        )
                        monitor_thread.start()
                        with _lock:
                            _active_trades[symbol] = trade_data
                            _monitoring_threads[symbol] = {
                                'thread': monitor_thread,
                                'stop_event': stop_event
                            }
                        logger.info(f"🔁 Recovered monitoring for {symbol}")
                    else:
                        remove_trade_from_redis(symbol)
                except Exception as e:
                    logger.debug(f"Failed to recover {symbol}: {e}")
                    remove_trade_from_redis(symbol)
    except Exception as e:
        logger.error(f"Trade recovery failed: {e}")

# if __name__ == "__main__":
#     recover_trades_on_startup()
#     start_external_close_watcher()

#     signal = {
#         "symbol": "AAVEUSDT",
#         "side": "SELL",
#         "entry": 196.67,
#         "take_profit": 190.00,
#         "stop_loss": 198.50,
#         "leverage": 10,
#         "confidence_percent": 90
#     }
#     print(f"🧪 Testing place_futures_order for {signal['symbol']}...")
#     place_futures_order(signal)