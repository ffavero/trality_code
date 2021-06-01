from datetime import datetime


"""
An attempt to reproduce the strategy 'h' as suggested
from Vezeris at al 2018, https://www.mdpi.com/1911-8074/11/3/56

NOTE:
The atr stop_loss/take_profit update at each cycle is
commented out as the order update does not work consistently
yet. Future fixes in trality backtest and order engine may
fix this. See 'FIXME' comment in the code
"""

INTERVAL = "1h"

SYMBOLS = [
   "ETHUSDT", "BTCUSDT", "ADAUSDT", "MATICUSDT"]

def initialize(state):
    state.number_offset_trades = 0;
    state.summary_performance = {}
    state.cooldown_price = {}
    state.cooldown_time = {}
    state.percent_invest = 0.98
    state.limit_orders = {}
    state.params = {}
    state.params["DEFAULT"] = {
        "max_hours_cooldown": 6,
        "atr_cooldown_n": 2,
        "atr_stop_loss_n": 3,
        "atr_take_profit_n": 5,
        "atr_period": 14,
        "macd_periods": [12, 26, 9]}


@schedule(interval=INTERVAL, symbol=SYMBOLS)
def handler(state, data):
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    buy_value = float(balance_quoted) * state.percent_invest
    n_pos = 0
    try:
        n_coins = len(data.keys())
        for this_symbol in data.keys():
            position = query_open_position_by_symbol(
                this_symbol, include_dust=False)
            if position is not None:
                n_pos += 1
        amount = 0
        if n_pos < n_coins:
            amount = buy_value / (n_coins - n_pos)
        for this_symbol in data.keys():
            handler_main(state, data[this_symbol], amount)
    except TypeError:
        handler_main(state, data, buy_value)
    if state.number_offset_trades < portfolio.number_of_offsetting_trades:
        
        pnl = query_portfolio_pnl()
        offset_trades = portfolio.number_of_offsetting_trades
        number_winners = portfolio.number_of_winning_trades
        msg_data = {"pnl": str(pnl), "number_winners": number_winners,
            "offset_trades": offset_trades, "best_trade_return": portfolio.best_trade_return * 100,
            "worst_trade_return": portfolio.worst_trade_return * 100,
            "average_profit_per_winning_trade": portfolio.average_profit_per_winning_trade,
            "average_loss_per_losing_trade": portfolio.average_loss_per_losing_trade}
        msg = (
            "-------\n"
            "Accumulated Pnl of Strategy:\n   %(pnl)s\n"
            "Number of winning trades %(number_winners)i / %(offset_trades)i.\n"
            "Best trade Return : %(best_trade_return).2f%%\n"
            "Worst trade Return : %(worst_trade_return).2f%%\n"
            "Average Profit per Winning Trade : %(average_profit_per_winning_trade).2f\n"
            "Average Loss per Losing Trade : %(average_loss_per_losing_trade).2f\n")
        print(msg % msg_data)
        # reset number offset trades
        state.number_offset_trades = portfolio.number_of_offsetting_trades



def handler_main(state, data, amount):
    symbol = data.symbol
    buy_value = amount
    params = get_default_params(state, symbol)

    atr_cooldown_n = params["atr_cooldown_n"]
    atr_stop_loss_n = params["atr_stop_loss_n"]
    atr_take_profit_n = params["atr_take_profit_n"]
    macd_periods = params["macd_periods"]
    atr_period = params["atr_period"]
    max_hours_cooldown = params["max_hours_cooldown"]


    position = query_open_position_by_symbol(
        symbol, include_dust=False)

    has_position = position is not None

    try:
        summary_perf = state.summary_performance[symbol]
    except KeyError:
        state.summary_performance[symbol] = {
            "positions": [], "winning": 0, "tot": 0, "pnl": 0}
        summary_perf = state.summary_performance[symbol]
    if has_position and position.id not in summary_perf["positions"]:
        state.summary_performance[symbol]["positions"].append(position.id)
    if int(datetime.fromtimestamp(data.last_time / 1000).minute) == 0:
        position_ids = state.summary_performance[symbol]['positions']
        still_open_positions = []
        for pos_id in position_ids:
            position = query_position_by_id(pos_id)
            if position.exit_price is None:
                still_open_positions.append(pos_id)
            else:
                pnl = float(position.realized_pnl)
                if pnl > 0:
                    state.summary_performance[symbol]['winning'] += 1
                state.summary_performance[symbol]['tot'] += 1
                state.summary_performance[symbol]['pnl'] += pnl
        state.summary_performance[symbol]['positions'] = still_open_positions
        perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
        print(
            perf_message % (
                symbol, state.summary_performance[symbol]['winning'],
                state.summary_performance[symbol]['tot'],
                float(state.summary_performance[symbol]['pnl'])))
    macd = data.macd(macd_periods[0], macd_periods[1], macd_periods[2])
    atr = data.atr(atr_period).last
    current_price = data.close_last
    current_low = data.low_last

    if macd is None:
        return

    stop_loss, sl_price = atr_tp_sl_percent(
        float(current_price), float(atr), atr_stop_loss_n, False)
    take_profit, tp_price = atr_tp_sl_percent(
        float(current_price), float(atr), atr_take_profit_n, True)


    buy_signal = macd.select("macd_histogram")[-1] > 0
    sell_signal = macd.select("macd_histogram")[-1] < 0

    if has_position and not sell_signal:
        # FIXME: This part results in no stop loss/take profit
        #        triggering at all in backtest, and saldom
        #        triggering in live bots

        # position_price = float(position.entry_price)
        # new_sl, sl_price = atr_tp_sl_percent(
        #     float(position_price), float(atr), stop_loss_n, False)
        # new_tp, tp_price = atr_tp_sl_percent(
        #     float(position_price), float(atr), take_profit_n, True)
        # cancel_state_limit_orders(state, symbol)
        # make_double_barrier(
        #      symbol, float(position.exposure), new_tp,
        #      new_sl, state)
        try:
            tp_price = state.limit_orders[
                symbol]["order_upper"].stop_price
            sl_price = state.limit_orders[
                symbol]["order_lower"].stop_price
            with PlotScope.root(symbol):
                plot("tp", tp_price)
                plot("sl", sl_price)
        except Exception:
            pass
    # cooldown checks after stop loss has been triggered
    try:
        cooldown_price = state.cooldown_price[symbol]
    except KeyError:
        state.cooldown_price[symbol] = None
        cooldown_price = state.cooldown_price[symbol]
    if cooldown_price is None:
        try:
            sl_order = state.limit_orders[symbol]["order_lower"]
            stop_price = sl_order.stop_price
            if current_low <= stop_price:
                cooldown_up_percent, cooldown_price = atr_tp_sl_percent(
                    float(stop_price), float(atr), atr_cooldown_n, True)
            state.cooldown_price[symbol] = cooldown_price
            state.cooldown_time[symbol] = data.last_time / 1000
        except Exception:
            pass

    if cooldown_price is not None:
        sl_time = datetime.fromtimestamp(state.cooldown_time[symbol])
        this_time = datetime.fromtimestamp(data.last_time / 1000)
        hours_elapsed_cooldown = (this_time - sl_time).seconds // 3600
        if cooldown_price < current_price or hours_elapsed_cooldown >= max_hours_cooldown:
            cooldown = False
        else:
            print("%s cooldown until price reach %s or in %i hour(s)" % (
                symbol, cooldown_price, max_hours_cooldown - hours_elapsed_cooldown))
            cooldown = True
    else:
        cooldown = False

    if buy_signal and cooldown is False and not has_position:
        signal_msg_data = {"symbol": symbol, "value": buy_value, "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Buy Signal: creating market order for %(symbol)s\n"
            "Buy value: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        print(signal_msg % signal_msg_data)
        buy_order = order_market_value(
            symbol=symbol, value=buy_value)
        make_double_barrier(
            symbol, float(buy_order.quantity), take_profit,
            stop_loss, state)
    elif sell_signal and has_position:
        signal_msg_data = {"symbol": symbol, "amount": position.exposure, "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Sell Signal: creating market order for %(symbol)s\n"
            "Sell amount: %(amount)s at current market price %(current_price)f\n"
            "++++++\n")
        print(signal_msg % signal_msg_data)
        cancel_state_limit_orders(state, symbol)
        close_position(symbol)


def atr_tp_sl_percent(close, atr, n=6, tp=True):
    if tp is True:
        tp = close + (n * atr)
    else:
        tp = close - (n * atr)
    return (abs(tp - close) / close, tp)


def make_double_barrier(symbol,amount,take_profit,stop_loss,state):

    """make_double_barrier

    This function creates two iftouched market orders with the onecancelsother
    scope. It is used for our tripple-barrier-method

    Args:
        amount (float): units in base currency to sell
        take_profit (float): take-profit percent
        stop_loss (float): stop-loss percent
        state (state object): the state object of the handler function
    
    Returns:
        TralityOrder:  two order objects
    
    """

    with OrderScope.one_cancels_others():
        order_upper = order_take_profit(symbol,amount,
        								take_profit,
                                        subtract_fees=True)
        order_lower = order_stop_loss(symbol,amount,
        							  stop_loss,
                                      subtract_fees=True)
        
    if order_upper.status != OrderStatus.Pending:
        errmsg = "make_double barrier failed with: {}"
        raise ValueError(errmsg.format(order_upper.error))
    
    # saving orders


    try:
        state.limit_orders[symbol]
        state.limit_orders[symbol]["order_upper"] = order_upper
        state.limit_orders[symbol]["order_lower"] = order_lower
        state.limit_orders[symbol]["created_time"] = order_upper.created_time
    except Exception:
        state.limit_orders[symbol] = {}
        state.limit_orders[symbol]["order_upper"] = order_upper
        state.limit_orders[symbol]["order_lower"] = order_lower
        state.limit_orders[symbol]["created_time"] = order_upper.created_time


    return order_upper, order_lower


def cancel_state_limit_orders(state, symbol):
    try:
        cancel_order(state.limit_orders[symbol]['order_lower'].id)
        state.limit_orders[symbol]['order_lower'] = None
    except Exception:
        pass
    try:
        cancel_order(state.limit_orders[symbol]['order_upper'].id)
        state.limit_orders[symbol]['order_upper'] = None
    except Exception:
        pass

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
