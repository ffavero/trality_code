##+------------------------------------------------------------------+
##| CCI BBANDS | 15m                                             |
##+------------------------------------------------------------------+

SYMBOLS_1 = "BTCUSDT"
SYMBOLS1 = ["VITEUSDT", "MATICUSDT", "RUNEUSDT", "ZILUSDT", "1INCHUSDT"]
SYMBOLS2 = ["MANAUSDT", "IRISUSDT", "GTOUSDT"]
SYMBOLS3 = ["LUNAUSDT", "NKNUSDT", "NEOUSDT"]
#SYMBOLS2 = ["MIRUSDT", "ZRXUSDT", "MANAUSDT", "CLVUSDT", "ALGOUSDT", "BNBUSDT"]
SYMBOLS1 = ["MATICUSDT"]
SYMBOLS2 = ["ZRXUSDT"]
SYMBOLS3 = ["MANAUSDT", "BNBUSDT"]


AUTHOR = "Francesco @79bass 2022-04-22"
DONATE = ("TIP JAR WALLET:  " +
          "BEP-20: 0xc7F0A80f8a16F50067ABcd511f72a6D4eeAFC59c"
          "ERC20:  0xc7F0A80f8a16F50067ABcd511f72a6D4eeAFC59c")


INTERVAL = "15m"

N_SYMBOLS = 6                   # Define how many symbols we are trading
                                # in all handlers combined

LEVERAGE =  1                   # Multiply the amount for a given number
                                # eg balance is 1000 usdt, N_SYMBOLS is 10
                                # LEVERAGE is 2; instead of 100 usdt per trade
                                # the bot will use 200 usdt (if enough balance
                                # is available)

FIX_BUY_AMOUNT = 150           # Specify a fix amount for trade if needed
                                # This will override any preference set by the
                                # N_SYMBOLS and LEVERAGE options



import numpy as np
from numpy import greater, less, sum, nan_to_num, exp
from datetime import datetime

##+------------------------------------------------------------------+
##| Settings in state (could set different tp/sl for each symbol)    |
##+------------------------------------------------------------------+

def initialize(state):
    state.number_offset_trades = 0
    state.params = {}
    state.past_daily_candles = {}
    state.hourly_candles = {}
    state.balance_quoted = 0
    state.params["DEFAULT"] = {
        "bolligner_period": 20,
        "bolligner_sd": 2,
        "keltner_ema": 20,
        "keltner_atr": 20,
        "keltner_n": 2,
        "keep_signal": 20}


##+------------------------------------------------------------------+
##| SYMBOL                                                           |
##+------------------------------------------------------------------+

@schedule(interval=INTERVAL, symbol=SYMBOLS1, window_size=200)
def handler1(state, data):
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    state.balance_quoted = float(balance_quoted)
    if FIX_BUY_AMOUNT is None:
        buy_value = float(portfolio.portfolio_value) / N_SYMBOLS * LEVERAGE
    else:
        buy_value = FIX_BUY_AMOUNT
    n_pos = 0
    try:
        for this_symbol in data.keys():
            handler_main(state, data[this_symbol], buy_value)
    except TypeError:
        handler_main(state, data, buy_value)


##+------------------------------------------------------------------+
##| SYMBOL2                                                          |
##+------------------------------------------------------------------+


@schedule(interval=INTERVAL, symbol=SYMBOLS3, window_size=200)
def handler2(state, data):
    portfolio = query_portfolio()
    if FIX_BUY_AMOUNT is None:
        buy_value = float(portfolio.portfolio_value) / N_SYMBOLS * LEVERAGE
    else:
        buy_value = FIX_BUY_AMOUNT
    n_pos = 0
    try:
        for this_symbol in data.keys():
            handler_main(state, data[this_symbol], buy_value)
    except TypeError:
        handler_main(state, data, buy_value)

@schedule(interval=INTERVAL, symbol=SYMBOLS2, window_size=200)
def handler3(state, data):
    portfolio = query_portfolio()
    if FIX_BUY_AMOUNT is None:
        buy_value = float(portfolio.portfolio_value) / N_SYMBOLS * LEVERAGE
    else:
        buy_value = FIX_BUY_AMOUNT
    n_pos = 0
    try:
        for this_symbol in data.keys():
            handler_main(state, data[this_symbol], buy_value)
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

#### Waiting Functions



def cross_over_keltner_buy(position_manager, trade_data, indicators_data):  
    signal = False
    kcl = indicators_data["keltner"]["lower"]
    price = indicators_data["close"]["data"]
    if price[-1] > kcl[-1] and price[-2] < kcl[-2]:
        signal = True
        position_manager.stop_waiting()
    return signal   

def signal_no_wait(position_manager, trade_data, indicators_data):  
    position_manager.stop_waiting()
    return True

####

def handler_main(state, data, amount):
    if data is None:
        return

    symbol = data.symbol

    if symbol == "BNBUSDT":
        refill_bnb(state, data, 20)
        return

    #--------------------------------------------#
    # Get Parameters and init variables in state #
    #--------------------------------------------#

    params = get_default_params(state, symbol)
    bolligner_period = params["bolligner_period"]
    bolligner_sd = params["bolligner_sd"]
    keltner_ema = params["keltner_ema"]
    keltner_atr = params["keltner_atr"]
    keltner_n = params["keltner_n"]
    keep_signal = params["keep_signal"]


    try:
        past_daily_candles = state.past_daily_candles[symbol]
    except KeyError:
        state.past_daily_candles[symbol] = {
            "yesterday": None, "2yesterday": None}
        past_daily_candles = state.past_daily_candles[symbol]

    try:
        hourly_candles = state.hourly_candles[symbol]
    except KeyError:
        state.hourly_candles[symbol] = []
        hourly_candles = state.hourly_candles[symbol]


    #------------#
    # Indicators #
    #------------#


    cci_data = data.cci(20)
    cci = cci_data.last
    adx_data = data.adx(14)
    adx = adx_data.last
    atr_data = data.atr(12)
    atr = atr_data.last


    yesterday_candle = past_daily_candles["yesterday"]
    before_yesterday_candle = past_daily_candles["2yesterday"]

    if yesterday_candle is None:
        past_daily_candles["yesterday"] = get_yesterday_daily_candle(data)
        state.past_daily_candles[symbol] = past_daily_candles
        yesterday_candle = past_daily_candles["yesterday"]

    if int(datetime.fromtimestamp(data.last_time / 1000).minute) == 0:
            if int(datetime.fromtimestamp(data.last_time / 1000).hour) == 0:
                past_daily_candles["2yesterday"] = past_daily_candles["yesterday"]
                past_daily_candles["yesterday"] = get_yesterday_daily_candle(data)
                state.past_daily_candles[symbol] = past_daily_candles

    yesterday_levels = compute_daily_levels(yesterday_candle)

    r1 = yesterday_levels["resistance1"]
    s1 = yesterday_levels["support1"]
    pivot = yesterday_levels["pivot"]

    # yesterday_levels_cmr = compute_daily_levels_camarilla(yesterday_candle)
    # r1_cmr = yesterday_levels_cmr["resistance1"]
    # s1_cmr = yesterday_levels_cmr["support1"]
    # r2_cmr = yesterday_levels_cmr["resistance2"]
    # s2_cmr = yesterday_levels_cmr["support2"]
    # r3_cmr = yesterday_levels_cmr["resistance3"]
    # s3_cmr = yesterday_levels_cmr["support3"]
    # r4_cmr = yesterday_levels_cmr["resistance4"]
    # s4_cmr = yesterday_levels_cmr["support4"]

    if before_yesterday_candle:
        before_yesterday_levels = compute_daily_levels(before_yesterday_candle)
        past_r1 = before_yesterday_levels["resistance1"]
        past_s1 = before_yesterday_levels["support1"]
    else:
        past_r1 = None
        past_s1 = None

    take_last = 150
    max_1h_candles = 500

    bbands = data.bbands(
        bolligner_period, bolligner_sd)
    kbands = keltner_channels(
        data, keltner_ema, keltner_atr, keltner_n, take_last)

    current_price = data.close_last
    current_low = data.low_last



    last_closes = data.close.select("close")[-take_last:]
    last_lows = data.low.select("low")[-take_last:]
    last_ccis = cci_data.select("cci")[-take_last:]
    last_adxs = adx_data.select("dx")[-take_last:]


    bbands_above_keltner_up = bbands.select(
        'bbands_upper')[-1] > kbands['high'][-1]
    bbands_below_keltner_low = bbands.select(
        'bbands_lower')[-1] < kbands['low'][-1] 

    p_lows, cci_lows = bbands_levels(
        last_ccis, last_lows, last_closes, bbands.select(
            'bbands_lower')[-take_last:])



    #-----------------------------#
    # Collect info and indicators #
    #-----------------------------#

    indicators_data = {
        "adx": {
            "data": last_adxs.tolist()[-5:]
        },
        "cci": {
            "data": last_ccis.tolist()[-5:]
        },
        "close": {
            "data": last_closes.tolist()[-5:]
        },
        "low": {
            "data": last_lows.tolist()[-5:]
        },
        "bollinger": {
            "upper": bbands.select("bbands_upper")[-5:], 
            "middle":bbands.select("bbands_middle")[-5:], 
            "lower": bbands.select("bbands_lower")[-5:]
        },
        "keltner": {
            "upper": kbands["high"][-5:], 
            "middle":kbands["middle"][-5:], 
            "lower": kbands["low"][-5:]
        },
    }
    

    #--------------------------#
    # Init the PositionManager #
    #--------------------------#

    position_manager = PositionManager(state, data.symbol, data.last_time)
    balance_quoted = state.balance_quoted
    position_manager.set_value(float(amount), update=True)

    #-------------------------------------#
    # Assess Stop loss/take profit limits #
    #-------------------------------------#



    # stop_loss = 0.05
    take_profit = 0.5


    current_bottom =  min(last_lows[-5:])
    sl_offset_percent = 0.02
    
    """Place stop loss for manually added positions"""
    if position_manager.has_position:
        if not position_manager.is_stop_placed():
            if past_s1:
                stop_loss_price = min([past_s1, s1, current_bottom])
            else:
                stop_loss_price = min([s1, current_bottom])

            stop_loss_price = stop_loss_price - (
                stop_loss_price * sl_offset_percent)
            stop_loss = price_to_percent(current_price, stop_loss_price)
            position_manager.double_barrier(take_profit, stop_loss)
        else:
            if past_s1:
                stop_loss_price = min([past_s1, s1])
            else:
                stop_loss_price = s1
    else:
        if past_s1:
            stop_loss_price = min([past_s1, s1, current_bottom])
        else:
            stop_loss_price = min([s1, current_bottom])        
    stop_loss_price = stop_loss_price - (
        stop_loss_price * sl_offset_percent)
    stop_loss = price_to_percent(current_price, stop_loss_price)


    """
    Check entry and stop loss values
    """

    try:
        sl_price = position_manager.position_data[
            "stop_orders"]["order_lower"].stop_price
    except Exception:
         sl_price = None
    entry_price = position_manager.get_entry_price()

    """
    Lift the stop loss when passed a support/resistance line
    """

    if position_manager.has_position and sl_price:
        if s1 < current_price:
            if sl_price < stop_loss_price:
                position_manager.update_double_barrier(
                    current_price,
                    stop_loss=stop_loss)

    #--------------------------------------------#
    # Feedback on PnL and data collection prints #
    #--------------------------------------------#

    if position_manager.pnl_changed:
        summary_performance = state.positions_manager[symbol]["summary"]
        perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
        print(
            perf_message % (
                symbol, summary_performance['winning'],
                summary_performance['tot'],
                float(summary_performance['pnl'])))
        perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
    if int(datetime.fromtimestamp(data.last_time / 1000).minute) == 0:
        if int(datetime.fromtimestamp(data.last_time / 1000).hour) == 0:
            summary_performance = state.positions_manager[symbol]["summary"]
            perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
            print(
                perf_message % (
                    symbol, summary_performance['winning'],
                    summary_performance['tot'],
                    float(summary_performance['pnl'])))
            perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")


    #----------------------#
    # Buy/Sell conditions  #
    #----------------------#


    buy_signal_wait = False
    sell_signal_wait = False

    if (current_price < pivot
        and past_s1 and s1 < past_s1
        and bbands_below_keltner_low
        and len(p_lows) > 1):
        if (
            last_closes[-1] > bbands.select("bbands_lower")[-1]
            and last_closes[-2] < bbands.select("bbands_lower")[-2]):
            if cci_lows[-1] > cci_lows[-2] and p_lows[-1] > p_lows[-2]:
                buy_signal_wait = True
    

    #----------------#
    # Resolve signal #
    #----------------#

    buy_signal = False
    sell_signal = False

    default_trade_data = {
        "signal_type": None,
        "status": None,
        "n_active": 0,
        "level": 0
    }

    """
    Start the wait/check/stop process
    """
    if position_manager.check_if_waiting():
        trade_data, trade_message = position_manager.waiting_data()
        trade_data["n_active"] += 1
        #if trade_data["n_active"] > keep_signal:
        if trade_data["status"] == "buying" and trade_data["n_active"] > keep_signal:
            """
            If the waiting eceeded the keep_signal limit
            reset the trade object and stop waiting
            """
            position_manager.stop_waiting()
            trade_message = None
            trade_data = default_trade_data

        if position_manager.has_position and trade_data["status"] == "buying":
            position_manager.stop_waiting()
            trade_message = None
            trade_data = default_trade_data
        elif not position_manager.has_position and trade_data["status"] == "selling":
            position_manager.stop_waiting()
            trade_message = None
            trade_data = default_trade_data
    else:
        trade_message = None
        trade_data = default_trade_data


    """
    Reset the trade data when a new signal pops up
    """
    if  not position_manager.has_position and buy_signal_wait :
        trade_data = default_trade_data
        trade_data["status"] = "buying"
        trade_data["signal_type"] = "cci_bband"
        position_manager.start_waiting(trade_data, "waiting to buy")
    elif position_manager.has_position and sell_signal_wait:
        trade_data = default_trade_data
        trade_data["status"] = "selling"
        trade_data["signal_type"] = "cci_bband"
        position_manager.start_waiting(trade_data, "waiting to sell")


    """
    define a dictionary with the confirmation function
    """
    confirmation_functions = {
        "buy": {
            "signal_cci_bband": cross_over_keltner_buy,
        },
        "sell": {
            "signal_cci_bband": signal_no_wait,
        }
    }
    """
    If the position is waiting we need to check for confirmation
    """
    if position_manager.check_if_waiting():
        if trade_data["status"] == "buying":
            buy_signal = confirmation_functions["buy"]["signal_%s" %
                trade_data["signal_type"]](
                    position_manager, trade_data, indicators_data)
        elif trade_data["status"] == "selling":
            sell_signal = confirmation_functions["sell"]["signal_%s" %
                trade_data["signal_type"]](
                    position_manager, trade_data, indicators_data)


    #-------------------------------------------------#
    # Assess available balance and target trade value #
    #-------------------------------------------------#

    skip_buy = False
    if balance_quoted <= position_manager.position_value and not position_manager.has_position:
        if balance_quoted < 20:
            print(
                "WARNING: Balance quoted (%s) is less than "
                "the minimum buy amount (%s)." % (
                    balance_quoted, position_manager.position_value))
            skip_buy = True
        else:
            position_manager.set_value(
                balance_quoted * 0.95, update=True)

    #--------------#
    # Plot section #
    #--------------#

    with PlotScope.root(symbol):
        plot("k_ema", kbands["middle"][-1])
        plot("k_upper", kbands["high"][-1])
        plot("k_lower", kbands["low"][-1])

    with PlotScope.root(symbol):
        plot("daily_resistance", r1)
        plot("daily_support", s1)
        plot("pivot", pivot)


    try:
        tp_price = position_manager.position_data[
            "stop_orders"]["order_upper"].stop_price
        sl_price = position_manager.position_data[
            "stop_orders"]["order_lower"].stop_price
        with PlotScope.root(position_manager.symbol):
            plot("tp", tp_price)
            plot("sl", sl_price)
    except Exception:
        pass


    with PlotScope.group("pnl", symbol):
        plot("pnl", float(state.positions_manager[
            symbol]["summary"]['pnl']))

    with PlotScope.group("cci_div", symbol):
        if len(cci_lows) > 1:
            plot("cci_inc", cci_lows[-1] > cci_lows[-2] )
            plot("price_inc", p_lows[-1] > p_lows[-2] )
    with PlotScope.group("buy", symbol):
        plot("buy_signal_wait", buy_signal_wait)


    #----------------------#
    # Buy/Sell instruction #
    #----------------------#

    if buy_signal and not position_manager.has_position:
        signal_msg_data = {
            "symbol": symbol,
            "value": position_manager.position_value,
            "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Buy Signal: creating market order for %(symbol)s\n"
            "Buy value: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        skip_msg = (
            "++++++\n"
            "Skip buy market order for %(symbol)s\n"
            "Not enough balance: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        # print(signal_msg % signal_msg_data)
        # position_manager.open_market()
        # position_manager.double_barrier(take_profit, stop_loss)

        if skip_buy is False:
            state.balance_quoted -= position_manager.position_value
            position_manager.open_market()
            position_manager.double_barrier(take_profit, stop_loss)
            print(signal_msg % signal_msg_data)
        else:
            print(skip_msg % signal_msg_data)


    elif sell_signal and position_manager.has_position:
        signal_msg_data = {
            "symbol": symbol,
            "amount": position_manager.position_exposure(),
            "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Sell Signal: creating market order for %(symbol)s\n"
            "Sell amount: %(amount)s at current market price %(current_price)f\n"
            "++++++\n")
        print(signal_msg % signal_msg_data)
        position_manager.close_market()


##+------------------------------------------------------------------+
##| methods and helpers                                              |
##+------------------------------------------------------------------+

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


def cross_over(x, y):
    if y[1] < x[1]:
        return False
    else:
        if x[0] > y[0]:
            return True
        else:
            return False


def cross_under(x, y):
    if y[1] > x[1]:
        return False
    else:
        if x[0] < y[0]:
            return True
        else:
            return False


def price_to_percent(close, price):
    return abs(price - close) / close

def atr_tp_sl_percent(close, atr, n=6, tp=True):
    if tp is True:
        tp = close + (n * atr)
    else:
        tp = close - (n * atr)
    return (price_to_percent(close, tp), tp)


class PositionManager:
    """
    A simple helper to manage positions boilerplate functionalities.
    It wraps around and extends the Trality position functionalities.
        - Query for open position (query_open_position_by_symbol)
        - Store the orders relative to the current position in state
          - A position can be declared without the orders to be filled/placed yet:
            waiting confirmation (TODO) or waiting filling limit orders (TODO)
          - Stop orders can be declared by the same class that opens the position
            getting all the info and storing the stop orders objects with the
            current corresponding symbol-position with the other orders
        - Keep track of per-symbols pnl and winning/losing records (extending the
          base position implementation where it's impossible to record pnl of a position
          terminated by a stop order before the first candle)
    propose syntax:
        position_manager = PositionManager(state, "BTCUSDT")
    query if has position:
        position_manager.has_position
    Set a value to the position
        position_manager.set_value(position_value)
    open the position:
        position_manager.open_market()
    save things in state without placing orders (eg waiting for confirmation):
        position_manager.open_wait_confirmation()
        # this will set True to the attribute position_manager.is_pending 
    open a position with a limit order and deal with the pending status:
        position_manager.open_limit(price_limit)
        .... need to think a clean pattern for limit/if_touched/trailing orders
    check if stop orders are present and add them if not:
        if not position_manager.has_double_barrier:
            position_manager.double_barrier(
                stop_loss=0.1, take_profit=0.05)
    close the position:
        position_manager.close_market()
    """


    def __init__(self, state, symbol, timestamp, include_dust=False):
        position = query_open_position_by_symbol(
            symbol, include_dust=include_dust)
        self.symbol = symbol
        self.timestamp = int(timestamp)
        self.has_position = position is not None
        self.is_pending = False
        self.pnl_changed = False
        try:
            self.position_data = state.positions_manager[self.symbol]["data"]
        except TypeError:
            state.positions_manager = {}
            state.positions_manager[self.symbol] = {
                "data": self.default_data(),
                "summary": {
                    "last_closure_type": None, "last_pnl": 0, "winning": 0, "tot": 0, "pnl": 0}}
            self.position_data = state.positions_manager[self.symbol]["data"]
        except KeyError:
            state.positions_manager[self.symbol] = {
                "data": self.default_data(),
                "summary": {
                    "last_closure_type": None, "last_pnl": 0, "winning": 0, "tot": 0, "pnl": 0}}
            self.position_data = state.positions_manager[self.symbol]["data"]
        if self.has_position:
            self.position_data["position"] = position
            if self.position_data["buy_order"] is None:
                # Potential manual buy or existing positions
                # when the bot was started
                order_id = self.position_data["position"].order_ids[-1]
                self.position_data["buy_order"] = query_order(order_id)

        #TODO self.check_if_waiting()
        #TODO self.check_if_pending()
        if not self.has_position and not self.is_pending:
            if self.position_data["buy_order"] is not None:
                stop_orders_filled = self.is_stop_filled()
                if stop_orders_filled:
                    state.positions_manager[
                        self.symbol]["summary"][
                            "last_closure_type"] = stop_orders_filled["side"]
                else:
                    state.positions_manager[
                        self.symbol]["summary"][
                            "last_closure_type"] = "rule"                    
                try:
                    closed_position = self.position_data["position"]
                except KeyError:
                    closed_position = None
                if closed_position is not None:
                    closed_position = query_position_by_id(closed_position.id)
                    pnl = float(closed_position.realized_pnl)
                    if pnl > 0:
                        state.positions_manager[self.symbol]["summary"]["winning"] += 1
                    state.positions_manager[self.symbol]["summary"]["tot"] += 1
                    state.positions_manager[self.symbol]["summary"]["pnl"] += pnl
                    state.positions_manager[self.symbol]["summary"]["last_pnl"] = pnl
                    try:
                        if state.collect_data:
                            state.collect_data[
                                self.symbol][
                                    str(closed_position.entry_time)]["pnl"] = pnl
                    except KeyError:
                        pass
                    self.pnl_changed = True
                else:
                    if stop_orders_filled:
                        sold_value = float((
                            stop_orders_filled[
                                "order"].executed_quantity * stop_orders_filled[
                                    "order"].executed_price) - stop_orders_filled[
                                        "order"].fees)
                        pnl = sold_value - self.position_value()
                        if pnl > 0:
                            state.positions_manager[self.symbol]["summary"]["winning"] += 1
                        state.positions_manager[self.symbol]["summary"]["tot"] += 1
                        state.positions_manager[self.symbol]["summary"]["pnl"] += pnl
                        state.positions_manager[self.symbol]["summary"]["last_pnl"] = pnl
                        try:
                            if state.collect_data:
                                state.collect_data[
                                    self.symbol][str(
                                        stop_orders_filled["order"].created_time)]["pnl"] = pnl
                        except KeyError:
                            pass
                        self.pnl_changed = True

                # reset state and position data
                self.cancel_stop_orders()
                waiting_data = self.position_data["waiting"]
                state.positions_manager[self.symbol]["data"] = self.default_data()
                self.position_data = state.positions_manager[self.symbol]["data"]
                self.position_data["waiting"] = waiting_data
    
    def set_value(self, value, update=False):
        try:
            stored_value = self.position_data["value"]
        except KeyError:
            stored_value = None
        if stored_value is None:
           self.position_data["value"] = value 
           self.position_value = value
        else:
            self.position_value = stored_value
        if update:
            self.position_value = value

    def get_entry_price(self):
        entry_price = None
        if self.has_position:
            try:
                entry_price = float(
                    self.position_data["position"].entry_price)
            except Exception:
                pass
        return entry_price

    def open_market(self, add=False):
        try:
            buy_order = self.position_data["buy_order"]
        except KeyError:
            buy_order = None
        if buy_order is None:        
            buy_order = order_market_value(
                symbol=self.symbol, value=self.position_value)
            self.position_data["buy_order"] = buy_order
            #self.__update_state__()
        elif add == True:
            buy_order = order_market_value(
                symbol=self.symbol, value=self.position_value)
            self.position_data["buy_order"] = buy_order            
        else:
            print("Buy order already placed")
        # if self.check_if_waiting():
        #     self.stop_waiting()

    def close_market(self):
        if self.has_position:
            close_position(self.symbol)
            #amount = self.position_amount()
            #order_market_amount(self.symbol,-1 * subtract_order_fees(amount))
            self.cancel_stop_orders()
        # if self.check_if_waiting():
        #     self.stop_waiting()

    def double_barrier(self, take_profit, stop_loss, subtract_fees=False):
        try:
            stop_orders = self.position_data["stop_orders"]
        except KeyError:
            stop_orders = {
                "order_upper": None, "order_lower": None}
        if stop_orders["order_upper"] is None:
            amount = self.position_amount()
            #amount = self.position_exposure()
            if amount is None:
                print("No amount to sell in position")
                return
            with OrderScope.one_cancels_others():
                stop_orders["order_upper"] = order_take_profit(
                    self.symbol, amount, take_profit, subtract_fees=subtract_fees)
                stop_orders["order_lower"] = order_stop_loss(
                    self.symbol, amount, stop_loss, subtract_fees=subtract_fees)
            if stop_orders["order_upper"].status != OrderStatus.Pending:
                errmsg = "make_double barrier failed with: {}"
                raise ValueError(errmsg.format(stop_orders["order_upper"].error))
            self.position_data["stop_orders"] = stop_orders
        else:
            print("Stop orders already exist")

    def double_barrier_price(self, take_profit_price, stop_loss_price, subtract_fees=False):
        try:
            stop_orders = self.position_data["stop_orders"]
        except KeyError:
            stop_orders = {
                "order_upper": None, "order_lower": None}
        if stop_orders["order_upper"] is None:
            amount = subtract_order_fees(self.position_amount())
            #amount = self.position_exposure()
            if amount is None:
                print("No amount to sell in position")
                return
            with OrderScope.one_cancels_others():
                stop_orders["order_upper"] = order_iftouched_market_amount(
                    self.symbol, -1 * amount, take_profit_price)
                stop_orders["order_lower"] = order_iftouched_market_amount(
                    self.symbol, -1 * amount, stop_loss_price)
            if stop_orders["order_upper"].status != OrderStatus.Pending:
                errmsg = "make_double barrier failed with: {}"
                raise ValueError(errmsg.format(stop_orders["order_upper"].error))
            self.position_data["stop_orders"] = stop_orders
        else:
            print("Stop orders already exist")

    def is_stop_filled(self):
        try:
            stop_orders = self.position_data["stop_orders"]
            stop_loss = stop_orders["order_lower"]
            take_profit = stop_orders["order_upper"]
        except KeyError:
            return None
        if stop_loss is not None:
            stop_loss.refresh()
            if stop_loss.is_filled():
                return {"side": "stop_loss", "order": stop_loss}
        if take_profit is not None:
            take_profit.refresh()
            if take_profit.is_filled():
                return {"side": "take_profit", "order": take_profit}

    def is_stop_placed(self):
        try:
            stop_orders = self.position_data["stop_orders"]
            stop_loss = stop_orders["order_lower"]
            take_profit = stop_orders["order_upper"]
        except KeyError:
            return False
        if stop_loss is None and take_profit is None:
            return False
        else:
            return True

    def update_double_barrier(self, current_price, take_profit=None, stop_loss=None, subtract_fees=False):
        success = True
        if take_profit is None:
            # keep upper as it is
            try:
                order_upper_price = float(self.position_data[
                    "stop_orders"]["order_upper"].stop_price)
                take_profit = abs(
                    order_upper_price - current_price) / current_price
            except:
                success = False
        if stop_loss is None:
            # Keep low as it is
            try:
                order_lower_price = float(self.position_data[
                    "stop_orders"]["order_lower"].stop_price)
                stop_loss = abs(
                    order_lower_price - current_price) / current_price
            except:
                success = False
        if success:
            self.cancel_stop_orders()
            self.double_barrier(
                take_profit, stop_loss, subtract_fees=subtract_fees)
        else:
            print("update stop limits failed")

    def update_double_barrier_price(self, current_price, take_profit_price=None, stop_loss_price=None, subtract_fees=False):
        success = True
        if take_profit_price is None:
            # keep upper as it is
            try:
                take_profit_price = float(self.position_data[
                    "stop_orders"]["order_upper"].stop_price)
            except:
                success = False
        if stop_loss_price is None:
            # Keep low as it is
            try:
                stop_loss_price = float(self.position_data[
                    "stop_orders"]["order_lower"].stop_price)
            except:
                success = False
        if success:
            self.cancel_stop_orders()
            self.double_barrier_price(
                take_profit_price, stop_loss_price, subtract_fees=subtract_fees)
        else:
            print("update stop limits failed")

    def position_amount(self):
        try:
            amount = float(self.position_data["buy_order"].quantity)
        except Exception:
            amount = None
        return amount

    def position_value(self):
        try:
            buy_order = self.position_data["buy_order"]
            buy_order.refresh()
            value = float(
                (buy_order.executed_quantity * buy_order.executed_price) - buy_order.fees)
        except KeyError:
            value = None
        return value

    def position_exposure(self):
        try:
            exposure = float(self.position_data["position"].exposure)
        except KeyError:
            exposure = None
        return exposure       
    
    def cancel_stop_orders(self):
        try:
            stop_orders = self.position_data["stop_orders"]
        except KeyError:
            stop_orders = {
                "order_upper": None, "order_lower": None}
        for stop_level in stop_orders:
            if stop_orders[stop_level] is not None:
                try:
                    cancel_order(stop_orders[stop_level].id)
                    stop_orders[stop_level] = None
                except Exception:
                    pass
        self.position_data["stop_orders"] = stop_orders
    
    def collect_data(self, state, data_dict):
        if state.collect_data is None:
            state.collect_data = {}
        try:
            state.collect_data[self.symbol]["%s" % self.timestamp] = data_dict
        except KeyError:
            state.collect_data[self.symbol] = {}
            state.collect_data[self.symbol]["%s" % self.timestamp] = data_dict        
    
    def start_waiting(self, waiting_data=None, waiting_message=None):
        if waiting_data:
            self.position_data["waiting"]["data"] = waiting_data
        if waiting_message:
            self.position_data["waiting"]["message"] = waiting_message
        self.position_data["waiting"]["status"] = True

    def stop_waiting(self, waiting_data=None, waiting_message=None):
        self.position_data["waiting"]["status"] = False
        self.position_data["waiting"]["data"] = waiting_data
        self.position_data["waiting"]["message"] = waiting_message

    def check_if_waiting(self):
        if self.position_data["waiting"]["status"] is None:
            return False
        else:
            return self.position_data["waiting"]["status"]

    def waiting_data(self):
        return (
            self.position_data["waiting"]["data"],
            self.position_data["waiting"]["message"])

    def default_data(self):
        return {
            "stop_orders": {"order_upper": None, "order_lower": None},
            "position": None,
            "waiting": {"status": None, "data": None, "message": None},
            "buy_order": None,
            "value": None
        }


def keltner_channels(data, period=20, atr_period=10, kc_mult=2, take_last=50):
    """
    calculate keltner channels mid, up and low values
    """
    ema = data.close.ema(period).select('ema')
    atr = data.atr(atr_period).select('atr')
    high = ema[-take_last:] + (kc_mult * atr[-take_last:])
    low = ema[-take_last:] - (kc_mult * atr[-take_last:])
    return {'middle': ema, 'high': high, 'low': low}


# Marcos Duarte detect peaks
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] https://github.com/demotu/detecta/blob/master/docs/detect_peaks.ipynb
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    >>> detect_peaks(x, show=True, ax=axs[0], threshold=0.5, title=False)
    >>> detect_peaks(x, show=True, ax=axs[1], threshold=1.5, title=False)
    Version history
    ---------------
    '1.0.7':
        Part of the detecta module - https://pypi.org/project/detecta/  
    '1.0.6':
        Fix issue of when specifying ax object only the first plot was shown
        Add parameter to choose if a title is shown and input a title
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def get_yesterday_daily_candle(data):
    today = datetime.fromtimestamp(data.times[-1] / 1000).day
    yesterday = None
    op = None
    cl = None
    hi = None
    lo = None
    for i in range(len(data.times) - 1, -1, -1):
        day = datetime.fromtimestamp((data.times[i] / 1000)).day
        if day != today and yesterday is None:
            yesterday = datetime.fromtimestamp((data.times[i] / 1000)).day
            cl = float(data.close[i])
            hi = float(data.high[i])
            lo = float(data.low[i])
        elif yesterday is not None and day == yesterday:
            op = float(data.open[i])
            if float(data.low[i]) < lo:
                lo = float(data.low[i])
            if float(data.high[i]) > hi:
                hi = float(data.high[i])
        elif yesterday is not None and day != yesterday and day != today:
            #print({"today": today, "yesterday": yesterday, "last_data": day})
            return({"high": hi, "low": lo, "open": op, "close": cl})


def get_1h_candles(times, opens, closes, highs, lows, volumes):
    op = None
    cl = None
    hi = None
    lo = None
    vo = None
    for i in range(len(times)):
        minute = datetime.fromtimestamp((times[i] / 1000)).minute
        if minute == 15:
            op = float(opens[i])
            hi = float(highs[i])
            lo = float(lows[i])
            vo = float(volumes[i])
        elif vo and minute == 0:
            cl = float(closes[i])
            if float(lows[i]) < lo:
                lo = float(lows[i])
            if float(highs[i]) > hi:
                hi = float(highs[i])
            vo += float(volumes[i])
            yield({"high": hi, "low": lo, "open": op, "close": cl, "volume": vo})
            op = None
            cl = None
            hi = None
            lo = None
            vo = None
        elif vo:
            if float(lows[i]) < lo:
                lo = float(lows[i])
            if float(highs[i]) > hi:
                hi = float(highs[i])
            vo += float(volumes[i])


def compute_daily_levels(yesterday_candle):
    pp = (yesterday_candle[
        "high"] +yesterday_candle["low"] + yesterday_candle[
            "close"]) / 3
    resistance1 = 2 * pp - yesterday_candle["low"]
    support1 = 2 * pp - yesterday_candle["high"]
    resistance2 = pp + (yesterday_candle["high"] - yesterday_candle["low"])
    support2 = pp - (yesterday_candle["high"] -yesterday_candle["low"])
    resistance3 = pp + (2*(yesterday_candle["high"] - yesterday_candle["low"]))
    support3 = pp - (2*(yesterday_candle["high"] - yesterday_candle["low"]))
    return ({
        "pivot": pp,
        "resistance1": resistance1,
        "resistance2": resistance2,
        "resistance3": resistance3,
        "support1": support1,
        "support2": support2,
        "support3": support3
    })

def compute_daily_levels_camarilla(yesterday_candle):
    h_l = (
        yesterday_candle["high"] - yesterday_candle["low"]) * 1.1
    r4 = (h_l / 2) + yesterday_candle["close"]
    r3 = (h_l / 4) + yesterday_candle["close"]
    r2 = (h_l / 6) + yesterday_candle["close"]
    r1 = (h_l / 12) + yesterday_candle["close"]
    s1 = yesterday_candle["close"] - (h_l / 12)
    s2 = yesterday_candle["close"] - (h_l / 6)
    s3 = yesterday_candle["close"] - (h_l / 4)
    s4 = yesterday_candle["close"] - (h_l / 2)
    return ({
        "resistance1": r1,
        "resistance2": r2,
        "resistance3": r3,
        "resistance4": r4,
        "support1": s1,
        "support2": s2,
        "support3": s3,
        "support4": s4
    })


def refill_bnb(state, data, amount):
    position_manager = PositionManager(
        state, data.symbol, data.last_time)
    balance_quoted = state.balance_quoted
    position_manager.set_value(float(amount))
    low_bnb_amount = 5
    amount_bnb = 0
    portfolio = query_portfolio()
    if position_manager.has_position:
        amount_bnb = position_manager.position_exposure() * float(data.close_last)
    if balance_quoted >= amount and amount_bnb < low_bnb_amount:
        print(
            "Buying BNB, current amount %f" % amount_bnb)
        state.balance_quoted -= position_manager.position_value
        position_manager.open_market(add=True)


def bbands_levels(indicator, price, close, bband, crossover=True):
    """
    Return the price and the indicator value when a bandcrossing
    was observed

    :param indicator: the 1d np.array of the indicator
    :type indicator: np.array
    :param price: the 1d np.array of prices
    :type price: np.array
    :param close: the 1d np.array of close prices
    :type close: np.array
    :param bband: the 1d np.array of the band to test
    :type bband: np.array
    :param crossover: Test cross over, if false test cross under
    :type crossover: bool
    """
    if crossover:
        idx = np.where(np.diff(np.less(close, bband)) == 1)[0] + 1
    else:
        idx = np.where(np.diff(np.greater(close, bband)) == 1)[0] + 1
    
    idx_idx = np.where(np.diff(idx) > 1)[0] + 1
    idx_groups = np.split(idx, idx_idx)
    # v_min = np.vectorize(min_value, excluded=[0])
    # v_max = np.vectorize(max_value, excluded=[0])
    if crossover:
        price_val = [np.min(price[g]) for g in idx_groups if len(g) > 0]
        iind_val = [np.min(indicator[g]) for g in idx_groups if len(g) > 0]
    else:
        price_val = [np.max(price[g]) for g in idx_groups]
        iind_val = [np.max(indicator[g]) for g in idx_groups]
    return (price_val, iind_val)
