from datetime import datetime


"""
Attempt to make a very simple and stable grid bot for trality
"""

INTERVAL_BIG = "1h"
INTERVAL_SMALL = "1m"

SYMBOLS = [
   "BTCUSDT", "ETHUSDT", "ADAUSDT"]

def initialize(state):
    state.buy_orders = {}
    state.init_value = {}
    state.current_atr = {}
    state.current_price = {}
    state.params = {}
    state.semaphore = {}
    state.params["DEFAULT"] = {
        "max_hold_hours": 23,
        "max_open_orders": 4,
        "atr_grid_step": 1,
        "percent_grid_step": 0.015,
        "value_per_order": 20,
        "atr_period": 12}

@schedule(interval=INTERVAL_BIG, symbol=SYMBOLS)
def handler_big(state, data):
    try:
        n_coins = len(data.keys())
        for this_symbol in data.keys():
            handler_big_fun(state, data[this_symbol])
    except TypeError:
        handler_big_fun(state, data)

@schedule(interval=INTERVAL_SMALL, symbol=SYMBOLS)
def handler_small(state, data):
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    n_pos = 0
    try:
        n_coins = len(data.keys())
        for this_symbol in data.keys():
            handler_small_fun(state, data[this_symbol])
    except TypeError:
        handler_small_fun(state, data)


def handler_big_fun(state, data):
    symbol = data.symbol
    params = get_default_params(state, symbol)

    atr_period = params["atr_period"]

    atr = data.atr(atr_period).last
    current_price = data.close_last

    if atr is None:
        return
    try:
        fix_value = state.init_value[symbol]
    except KeyError:
        fix_value = None
        for low in data.low:
            if fix_value is None:
                fix_value = low
            elif low < fix_value:
                fix_value = low
        state.init_value[symbol] = fix_value
    state.current_atr[symbol] = atr
    state.current_price[symbol] = current_price


def handler_small_fun(state, data):
    symbol = data.symbol
    params = get_default_params(state, symbol)
    value_per_order = params["value_per_order"]
    atr_grid_step = params["atr_grid_step"]
    percent_grid_step = params["percent_grid_step"]
    max_open_orders = params["max_open_orders"]
    max_hold_hours = params["max_hold_hours"]

    try:
        atr = state.current_atr[symbol]
    except KeyError:
        state.current_atr[symbol] = None
        atr = None

    if atr is None:
        return
    try:
        fix_value = state.init_value[symbol]
    except KeyError:
        state.init_value[symbol] = None
        fix_value = None

    if fix_value is None:
        return

    try:
        big_candle_current_price = state.current_price[symbol]
    except KeyError:
        state.current_price[symbol] = None
        big_candle_current_price = None

    if big_candle_current_price is None:
        return

    current_price = data.close_last
    current_low = data.low_last
    current_high = data.high_last
    current_time = data.last_time

    if percent_grid_step is None:
        if atr_grid_step is None:
            raise(Exception(
                "At least one of the two options "
                "atr_grid_step or percent_grid_step need to be set"))
        percent_grid_step = atr_to_percent(
            float(fix_value), float(atr), atr_grid_step, False)
    n_step_from_origin = int(get_number_of_steps(
        current_price, fix_value, percent_grid_step))
    level_price = get_price_of_closest_step(
        fix_value, n_step_from_origin, percent_grid_step)
    #print("%s current price: %s level %i price: %s" % (
    #    symbol, current_price, n_step_from_origin, level_price))

    with PlotScope.root(symbol):
        for i in range(n_step_from_origin - 1, n_step_from_origin + 1):
            i = n_step_from_origin
            level_i_price = get_price_of_closest_step(
                fix_value, i, percent_grid_step)
            plot("%i" % i, level_i_price)

    try:
        buy_orders = state.buy_orders[symbol]
    except KeyError:
        state.buy_orders[symbol] = []
        buy_orders = []
    bought_levels = [x["level"] for x in buy_orders]
    if current_high >= level_price and current_price <= level_price:
        if len(buy_orders) < max_open_orders and n_step_from_origin not in bought_levels:
            print("%s Buying at level %i" % (symbol, n_step_from_origin))
            buy_order = order_market_value(symbol, value = value_per_order)
            state.buy_orders[symbol].append({
                "level": n_step_from_origin,
                "order": buy_order,
                "time": current_time
            })
    elif current_low <= level_price and current_price >= level_price:
        orders_left = []
        sell_amount = 0
        for order in buy_orders:
            order_level = order["level"]
            if order_level < n_step_from_origin:
                sell_amount += float(order["order"].quantity)
                print("%s Sell at level %i quantity bought at level %i" % (
                    symbol, n_step_from_origin, order_level))
            else:
                orders_left.append(order)
        if sell_amount > 0:
            print("%s %i orders left to sell" % (symbol, len(orders_left)))
            order_market_amount(symbol, -1 * subtract_order_fees(sell_amount))

        buy_orders = orders_left
    # sell old orders
    if max_hold_hours:
        orders_left = []
        sell_amount = 0
        for order in buy_orders:
            order_time = datetime.fromtimestamp(order["time"] / 1000)
            this_time = datetime.fromtimestamp(current_time / 1000)
            hold_hours = (this_time - order_time).seconds / 3600
            if hold_hours > max_hold_hours:
                sell_amount += float(order["order"].quantity)
                print("%s Sell at level %i: hours holding %s (max %s)" % (
                    symbol, n_step_from_origin, hold_hours, max_hold_hours))
            else:
                orders_left.append(order)
        if sell_amount > 0:
            print("%s %i orders left to sell" % (symbol, len(orders_left)))
            order_market_amount(symbol, -1 * subtract_order_fees(sell_amount))                
        buy_orders = orders_left
    state.buy_orders[symbol] = buy_orders

def get_number_of_steps(current_price, init_price, step_size):
    diff_percent = (current_price - init_price) / init_price
    return(int(round(diff_percent / step_size, 0)))

def get_price_of_closest_step(fix_value, n, step_size):
    return fix_value + (n * (fix_value * step_size))


def atr_to_percent(close, atr, n=6, tp=True):
    tp = close + (n * atr)
    return(abs(tp - close) / close)


def get_default_params(state, symbol):
    default_params = state.params["DEFAULT"]
    try:
        params = state.params[symbol]
        for key in default_params:
            if key not in params.keys():
                params[key] = default_params[key]
    except KeyError:
        params = default_params
    return params
