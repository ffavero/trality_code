from numpy import greater, less, sum, nan_to_num
import numpy as np
from datetime import datetime


TITLE = "Multicoin Bollinger Bands Bayesian Oscillator"
VERSION = "50.1"
ALIAS = "Macd_and_confirm"
AUTHOR = "Francesco @79bass 2021-06-18"
DONATE = ("TIP JAR WALLET:  "
          "BEP-20: 0xc7F0A80f8a16F50067ABcd511f72a6D4eeAFC59c"
          "ERC20:  0xc7F0A80f8a16F50067ABcd511f72a6D4eeAFC59c")

INTERVAL_MACD_COORDINATOR = "1h"

INTERVAL = "15m"
INTERVAL_BNB = "15m"

# INTERVAL_MACD_COORDINATOR = "6h"
# INTERVAL = "1h"
# INTERVAL_BNB = "1h"

SYMBOLS = [
    "VITEUSDT", "MATICUSDT", "RUNEUSDT", "ZILUSDT", "BTCDOWNUSDT"]
#"1INCHUSDT", "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ATOMUSDT"]
#"AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "BTTUSDT"]
#   "ADAUSDT", "MATICUSDT", "ETHUSDT", "DOTUSDT", "SOLUSDT"]
#   "ZILUSDT", "MATICUSDT", "RUNEUSDT", "VITEUSDT", "BTCDOWNUSDT"]

VERBOSE = 1

## 0 prints only signals and portfolio information
## 1 prints updating orders info
## 2 prints stats info at each candle

SIGNALS = [1, 5]




def initialize(state):
    state.number_offset_trades = 0
    state.percent_invest = 0.98
    state.bbres_prev = {}
    state.semaphore = {}
    state.zero_signal_timer = {}
    state.params = {}
    state.next_atr_price = {}
    state.params["DEFAULT"] = {
        "atr_stop_loss_n": 6,
        "atr_take_profit_n": 6,
        "lower_threshold": 15,
        "bayes_period": 20,
        "max_candels_with_0_signals": 24,
        "use_semaphore": True,
        "lenient_with_uptrend": True,
        "buy_with_ema45": False,
        "wait_for_confirmation": False,
        "filter_ema30": True,
        "signals_mode": SIGNALS}
    state.params["RUNEUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["VITEUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["ZILUSDT"] = {
        "signals_mode": [1, 3, 4, 5],
        "buy_with_ema45": False
    }

# @schedule(
#     interval=INTERVAL_MACD_COORDINATOR_LONG, symbol=SYMBOLS)
# def coordinator_long_handler(state, data):
#     try:
#         for this_symbol in data.keys():
#             coordinator_long(state, data[this_symbol])
#     except TypeError:
#         coordinator_long(state, data)


@schedule(
    interval=INTERVAL_MACD_COORDINATOR, symbol=SYMBOLS)
def coordinator_handler(state, data):
    try:
        for this_symbol in data.keys():
            coordinator_main(state, data[this_symbol])
    except TypeError:
        coordinator_main(state, data)


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

## handler to buy BNB to pay subtract_fees
@schedule(interval=INTERVAL_BNB, symbol="BNBUSDT")
def handler_bnb(state, data):
    add_bnb_amount = 11
    low_bnb_amount = 5
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    buy_value = float(balance_quoted) * state.percent_invest
    position = query_open_position_by_symbol(
        data.symbol, include_dust=True)
    has_position = position is not None
    n_pending = 0
    amount_bnb = 0
    if has_position:
        amount_bnb = position.position_value
    if buy_value >= add_bnb_amount and amount_bnb < low_bnb_amount:
        if n_pending == 0:
            print(
                "Buying BNB, current amount %f" % amount_bnb)
            order_market_value(data.symbol, value=add_bnb_amount)


def coordinator_main(state, data):
    if data is None:
        return
    symbol = data.symbol
    params = get_default_params(state, symbol)

    stop_loss_n = params["atr_stop_loss_n"]
    take_profit_n = params["atr_take_profit_n"]
    lenient_with_uptrend = params["lenient_with_uptrend"]
    try:
        last_semaphore = state.semaphore[symbol][1]
    except KeyError:
        state.semaphore[symbol] = ["green", "green"]
        last_semaphore = state.semaphore[symbol][1]
    macd = data.macd(
        12, 26, 9)
    vwma_data = data.vwma(14)
    adx = data.adx(14).last
    vwma = vwma_data.last
    vma_diff = vwma - vwma_data.select("vwma")[-2]
    current_price = data.close_last
    macd_histogram_last = macd.select("macd_histogram")[-1]
    macd_histogram_2nd_last = macd.select("macd_histogram")[-2]
    this_semaphore = state.semaphore[symbol][1]
    if macd_histogram_last < 0 or current_price < vwma:
        if lenient_with_uptrend:
            if vma_diff < 0:
                this_semaphore = "red"
        else:
            this_semaphore = "red"
    elif macd_histogram_last > 0 or current_price > vwma:
        this_semaphore = "green"
    state.semaphore[symbol][1] = this_semaphore


def handler_main(state, data, amount):
    if data is None:
        return
    symbol = data.symbol
    buy_value = amount
    params = get_default_params(state, symbol)

    stop_loss_n = params["atr_stop_loss_n"]
    take_profit_n = params["atr_take_profit_n"]
    lower_threshold = params["lower_threshold"]
    bayes_period =  params["bayes_period"]
    signals_mode =  params["signals_mode"]
    max_candels_with_0_signals = params["max_candels_with_0_signals"]
    use_semaphore = params["use_semaphore"]
    buy_with_ema45 = params["buy_with_ema45"]
    wait_for_confirmation = params["wait_for_confirmation"]
    filter_ema30 = params["filter_ema30"]
    atr_decrease_steps = 1

    try:
        semaphores = state.semaphore[symbol]
    except KeyError:
        state.semaphore[symbol] = ["green", "green"]
        semaphores = state.semaphore[symbol]
    try:
        zero_signal_timer = state.zero_signal_timer[symbol]
    except KeyError:
        state.zero_signal_timer[symbol] = 0
        zero_signal_timer = state.zero_signal_timer[symbol]
    try:
        next_atr_price = state.next_atr_price[symbol]
    except KeyError:
        state.next_atr_price[symbol] = [None, None]
        next_atr_price = state.next_atr_price[symbol]

    bb_period = 20
    bb_std_dev_mult = 2
    bbands = data.bbands(bb_period, bb_std_dev_mult)
    atr_data = data.atr(14)
    atr = atr_data.last
    adx_data = data.adx(14)
    adx = adx_data.last
    rsi = data.close.rsi(14).ema(5)
    rsi1 = data.close.rsi(4).ema(5)
    ema30 = data.ema(30).last
    cr45 = cross_over(data.ema(45).select("ema")[-2:], data.close.select("close")[-2:])
    cr45 = cr45 or cross_under(data.ema(45).select("ema")[-2:], data.close.select("close")[-2:])

    if bbands is None:
        return

    position_manager = PositionManager(state, symbol, data.last_time)
    position_manager.set_value(buy_value)

    if int(datetime.fromtimestamp(data.last_time / 1000).minute) == 0:
        summary_performance = state.positions_manager[symbol]["summary"]
        perf_message = ("%s winning positions %i/%i, realized pnl: %.3f")
        print(
            perf_message % (
                symbol, summary_performance['winning'],
                summary_performance['tot'],
                float(summary_performance['pnl'])))

    current_price = data.close_last
    stop_loss, sl_price = atr_tp_sl_percent(
        float(current_price), float(atr), stop_loss_n, False)
    take_profit, tp_price = atr_tp_sl_percent(
        float(current_price), float(atr), take_profit_n, True)
    
    # Place stop loss for manually added positions
    if position_manager.has_position and not position_manager.is_stop_placed():
        position_manager.double_barrier(take_profit, stop_loss)
        # next_atr_step = current_price + (atr_decrease_steps * atr)        
        # state.next_atr_price[symbol][0] = next_atr_step
        # state.next_atr_price[symbol][1] = stop_loss_n
        next_atr_step = current_price + (3 * atr)        
        state.next_atr_price[symbol][0] = next_atr_step
        state.next_atr_price[symbol][1] = 3

    # if position_manager.has_position and next_atr_price[0] < current_price:
    #     if next_atr_price[1] > 2:
    #         new_n = next_atr_price[1] - atr_decrease_steps
    #         next_step = current_price + (atr_decrease_steps * atr)
    #         new_stop_loss, new_stop_loss_price = atr_tp_sl_percent(
    #             float(current_price), float(atr), new_n, False)
    #         print("advance stop loss to %s atr" % new_n)
    #         position_manager.update_double_barrier(current_price, None, new_stop_loss)
    #         state.next_atr_price[symbol][0] = next_step
    #         state.next_atr_price[symbol][1] = new_n


    bb_res = bbbayes(
        data.close.select('close'), bayes_period,
        bbands.select('bbands_upper'), bbands.select('bbands_lower'),
        bbands.select('bbands_middle'))

    sigma_probs_up = bb_res[0]
    sigma_probs_down = bb_res[1]
    prob_prime = bb_res[2]

    if (sigma_probs_up + sigma_probs_down + prob_prime) == 0:
        state.zero_signal_timer[symbol] += 1
    else:
        state.zero_signal_timer[symbol] = 0
    try:
        bbres_prev = state.bbres_prev[symbol]
        trading_live = True
    except KeyError:
        state.bbres_prev[symbol] = [0, 0, 0]
        bbres_prev = state.bbres_prev[symbol]
        trading_live = False

    sigma_probs_up_prev = bbres_prev[0]
    sigma_probs_down_prev = bbres_prev[1]
    prob_prime_prev = bbres_prev[2]
    buy_signal = False
    sell_signal = False
    buy_signal_wait, sell_signal_wait, a, b = compute_signal(
        sigma_probs_up, sigma_probs_down, prob_prime,
        sigma_probs_up_prev, sigma_probs_down_prev,
        prob_prime_prev, lower_threshold, signals_mode)
    last_rsis = rsi1.select("ema")[-90:]

    last_adxs = adx_data.select("dx")[-90:]
    last_prices = data.select("close")[-90:]

    last_peaks = detect_peaks(
        last_rsis, mpd=8, mph=30, edge=None, kpsh=True)
    last_valleys = detect_peaks(
        last_rsis, mpd=8, mph=70, edge=None, valley=True, kpsh=True)
    last_dx_peaks = detect_peaks(last_adxs, mpd=8, edge=None)
    last_dx_valleys = detect_peaks(last_adxs, mpd=8, edge=None, valley=True)

    
    if wait_for_confirmation:
        """
        ## Strategy ideas note:
        ## Use RSI peaks data:
        ## if signal is right after a peak, compute:
        ##     - wich signal type (some signal may be promiscuous
        ##           some can only be buy/sell)
        ##        after identify promiscuous signal:
        ##        - if it's a valley buy
        ##        - if it's a peak sell
        ##     - Additionally evaluate previous peaks
        ##       compute divergence to evaluate future direction 
        ##           (and block/unblock the trade)

        if position_manager.has_position:
            # resolve sell signals
            sell_0 = get_signal_from_dict(0, b)
            sell_1 = get_signal_from_dict(1, b)
            sell_2 = get_signal_from_dict(2, b)
            sell_3 = get_signal_from_dict(3, b)
            sell_4 = get_signal_from_dict(4, b)
            sell_5 = get_signal_from_dict(5, b)

        else:
            # resolve buy signals
            buy_0 = get_signal_from_dict(0, a)
            buy_1 = get_signal_from_dict(1, a)
            buy_2 = get_signal_from_dict(2, a)
            buy_3 = get_signal_from_dict(3, a)
            buy_4 = get_signal_from_dict(4, a)
            buy_5 = get_signal_from_dict(5, a)

        """
    else:
        buy_signal = buy_signal_wait
        sell_signal = sell_signal_wait

    state.bbres_prev[symbol] = bb_res
    if buy_with_ema45:
        buy_signal = cr45
    if filter_ema30:
        if buy_signal and ema30 > bbands.select('bbands_middle')[-1]:
            pass
        else:
            buy_signal = False
    ## Plots sections

    with PlotScope.group("bayesian_bollinger", symbol):
        plot("sigma_up", bb_res[0])
        plot("sigma_down", bb_res[1])
        plot("prime_prob", bb_res[2])

    with PlotScope.group("rsi_cross", position_manager.symbol):
        plot("rsi_long", rsi.last)
        plot("rsi_short", rsi1.last)
        if len(last_peaks) > 0:
            plot("rsi_peaks", last_rsis[last_peaks][-1])
        if len(last_valleys) > 0:
            plot("rsi_valley", last_rsis[last_valleys][-1])

    with PlotScope.group("adx", position_manager.symbol):
        plot("adx", adx_data.select("dx")[-1])
        if len(last_dx_peaks) > 0:
            plot("adx_peaks", last_adxs[last_dx_peaks][-1])
        if len(last_dx_valleys) > 0:
            plot("adx_valley", last_adxs[last_dx_valleys][-1])

        

    with PlotScope.group("semaphore", symbol):
        if semaphores[1] == "red":
            plot("semaphore", 0)
        elif semaphores[1] == "green":
            plot("semaphore", 1)

    with PlotScope.group("signal", symbol):
        plot("0", int(a["0"]) + (-1 * int(b["0"])))
        try:
            plot("1", int(a["1"]) + (-1 * int(b["1"])))
        except KeyError:
            pass
        try:
            plot("2", int(a["2"]) + (-1 * int(b["2"])))
        except KeyError:
            pass
        try:
            plot("3", int(a["3"]) + (-1 * int(b["3"])))
        except KeyError:
            pass
        try:
            plot("4", int(a["4"]) + (-1 * int(b["4"])))
        except KeyError:
            pass
        try:
            plot("5", int(a["5"]) + (-1 * int(b["5"])))
        except KeyError:
            pass

    with PlotScope.group("signals", symbol):
        plot("buy_wait", int(buy_signal_wait))
        plot("sell_wait", -1 * int(sell_signal_wait))
        plot("buy", int(buy_signal))
        plot("sell", -1 * int(sell_signal))

    if position_manager.has_position:
        try:
            tp_price = position_manager.position_data[
                "stop_orders"]["order_upper"].stop_price
            sl_price = position_manager.position_data[
                "stop_orders"]["order_lower"].stop_price
            with PlotScope.root(symbol):
                plot("tp", tp_price)
                plot("sl", sl_price)
        except Exception:
            pass

    if (semaphores[1] == "red" or semaphores[0] == "red") and not position_manager.has_position and use_semaphore:
        semaphore_msg_close = (
            "The %s %s MACD histogram showing downtrend: "
            "stopping the strategy from buying "
            "until the trend changes")
        print(semaphore_msg_close % (
            symbol, INTERVAL_MACD_COORDINATOR))
        return

    if not trading_live:
        if VERBOSE >= 1:
            print("Skip first candle to gather signals")
        return
    if buy_signal and not position_manager.has_position:
        signal_msg_data = {"symbol": symbol, "value": buy_value, "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Buy Signal: creating market order for %(symbol)s\n"
            "Buy value: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        print(signal_msg % signal_msg_data)
        position_manager.open_market()
        position_manager.double_barrier(take_profit, stop_loss)
        # next_atr_step = current_price + (atr_decrease_steps * atr)        
        # state.next_atr_price[symbol][0] = next_atr_step
        # state.next_atr_price[symbol][1] = stop_loss_n
        next_atr_step = current_price + (3 * atr)        
        state.next_atr_price[symbol][0] = next_atr_step
        state.next_atr_price[symbol][1] = 3
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
        self.has_position = position is not None
        self.is_pending = False
        try:
            self.position_data = state.positions_manager[self.symbol]["data"]
        except TypeError:
            state.positions_manager = {}
            state.positions_manager[self.symbol] = {
                "data": self.default_data(),
                "summary": {
                    "positions": [], "winning": 0, "tot": 0, "pnl": 0}}
            self.position_data = state.positions_manager[self.symbol]["data"]
        except KeyError:
            state.positions_manager[self.symbol] = {
                "data": self.default_data(),
                "summary": {
                    "positions": [], "winning": 0, "tot": 0, "pnl": 0}}
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
        if not self.has_position and (not self.is_pending or not self.check_if_waiting()):
            if self.position_data["buy_order"] is not None:
                self.cancel_stop_orders()
                try:
                    closed_position = self.position_data["position"]
                except KeyError:
                    closed_position = None
                if closed_position is not None:
                    closed_position = query_position_by_id(closed_position.id)
                    pnl = float(closed_position.realized_pnl)
                    if pnl > 0:
                        state.positions_manager[self.symbol]["summary"]['winning'] += 1
                    state.positions_manager[self.symbol]["summary"]['tot'] += 1
                    state.positions_manager[self.symbol]["summary"]['pnl'] += pnl
                else:
                    stop_orders_filled = self.is_stop_filled()
                    if stop_orders_filled:
                        sold_value = float((
                            stop_orders_filled[
                                "order"].executed_quantity * stop_orders_filled[
                                    "order"].executed_price) - stop_orders_filled[
                                        "order"].fees)
                        pnl = sold_value - self.position_value()
                        if pnl > 0:
                            state.positions_manager[self.symbol]["summary"]['winning'] += 1
                        state.positions_manager[self.symbol]["summary"]['tot'] += 1
                        state.positions_manager[self.symbol]["summary"]['pnl'] += pnl
                # reset state and position data
                state.positions_manager[self.symbol]["data"] = self.default_data()
                self.position_data = state.positions_manager[self.symbol]["data"]
    
    def set_value(self, value):
        try:
            stored_value = self.position_data["value"]
        except KeyError:
            stored_value = None
        if stored_value is None:
           self.position_data["value"] = value 
           self.position_value = value
        else:
            self.position_value = stored_value

    def open_market(self):
        try:
            buy_order = self.position_data["buy_order"]
        except KeyError:
            buy_order = None
        if buy_order is None:        
            buy_order = order_market_value(
                symbol=self.symbol, value=self.position_value)
            self.position_data["buy_order"] = buy_order
            #self.__update_state__()
        else:
            print("Buy order already placed")
        if self.check_if_waiting():
            self.stop_waiting()

    def close_market(self):
        if self.has_position:
            close_position(self.symbol)
            self.cancel_stop_orders()
        if self.check_if_waiting():
            self.stop_waiting()

    def double_barrier(self, take_profit, stop_loss, subtract_fees=False):
        try:
            stop_orders = self.position_data["stop_orders"]
        except KeyError:
            stop_orders = {
                "order_upper": None, "order_lower": None}
        if stop_orders["order_upper"] is None:
            amount = self.position_amount()
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
        if take_profit is None:
            # keep upper as it is
            order_upper_price = float(self.position_data[
                "stop_orders"]["order_upper"].stop_price)
            take_profit = abs(order_upper_price - current_price) / current_price
        if stop_loss is None:
            # Keep low as it is
            order_lower_price = float(self.position_data[
                "stop_orders"]["order_lower"].stop_price)
            stop_loss = abs(order_lower__price - current_price) / current_price
        self.cancel_stop_orders()
        self.double_barrier(take_profit, stop_loss, subtract_fees=subtract_fees)

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
    
    def start_waiting(self, waiting_data=None, waiting_message=None):
        if waiting_data:
            self.position_data["waiting"]["data"] = waiting_data
        if waiting_message:
            self.position_data["waiting"]["message"] = waiting_message
        self.position_data["waiting"]["status"] = True

    def stop_waiting(self, waiting_data=None, waiting_message=None):
        self.position_data["waiting"]["status"] = False

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

def get_default_params(state, symbol):
    default_params = state.params["DEFAULT"]
    try:
        params = state.params[symbol]
        for key in default_params:
            if key not in params.keys():
                params[key] = default_params[key]
    except KeyError:
        params = default_params
        state.params[symbol] = params
    return params


def atr_tp_sl_percent(close, atr, n=6, tp=True):
    if tp is True:
        tp = close + (n * atr)
    else:
        tp = close - (n * atr)
    return (abs(tp - close) / close, tp)

def bbbayes(close, bayes_period, bb_upper, bb_basis, sma_values):
    prob_bb_upper_up_seq = greater(close[-bayes_period:],
    bb_upper[-bayes_period:])
    prob_bb_upper_down_seq = less(close[-bayes_period:],
    bb_upper[-bayes_period:])
    prob_bb_basis_up_seq = greater(close[-bayes_period:],
    bb_basis[-bayes_period:])
    prob_bb_basis_down_seq = less(close[-bayes_period:],
    bb_basis[-bayes_period:])
    prob_sma_up_seq = greater(close[-bayes_period:],
    sma_values[-bayes_period:])
    prob_sma_down_seq = less(close[-bayes_period:],
    sma_values[-bayes_period:])
    
    prob_bb_upper_up = sum(
        prob_bb_upper_up_seq) / bayes_period
    prob_bb_upper_down = sum(
        prob_bb_upper_down_seq) / bayes_period
    prob_up_bb_upper = prob_bb_upper_up / (prob_bb_upper_up + prob_bb_upper_down)
    prob_bb_basis_up = sum(
        prob_bb_basis_up_seq) / bayes_period
    prob_bb_basis_down = sum(
        prob_bb_basis_down_seq) / bayes_period
    prob_up_bb_basis = prob_bb_basis_up / (prob_bb_basis_up + prob_bb_basis_down)

    prob_sma_up = sum(
        prob_sma_up_seq) / bayes_period
    prob_sma_down = sum(
        prob_sma_down_seq) / bayes_period
    prob_up_sma = prob_sma_up / (prob_sma_up + prob_sma_down)

    sigma_probs_down = nan_to_num(
        prob_up_bb_upper * prob_up_bb_basis * prob_up_sma / prob_up_bb_upper * prob_up_bb_basis * prob_up_sma + (
            (1 - prob_up_bb_upper) * (1 - prob_up_bb_basis) * (
                1 - prob_up_sma)), 0)
    # Next candles are breaking Up
    prob_down_bb_upper = prob_bb_upper_down / (
        prob_bb_upper_down + prob_bb_upper_up)
    prob_down_bb_basis = prob_bb_basis_down / (
        prob_bb_basis_down + prob_bb_basis_up)
    prob_down_sma = prob_sma_down / (prob_sma_down + prob_sma_up)
    sigma_probs_up = nan_to_num(
        prob_down_bb_upper * prob_down_bb_basis * prob_down_sma / prob_down_bb_upper * prob_down_bb_basis * prob_down_sma + (
            (1 - prob_down_bb_upper) * (1 - prob_down_bb_basis) * (1 - prob_down_sma) ), 0)

    prob_prime = nan_to_num(
        sigma_probs_down * sigma_probs_up / sigma_probs_down * sigma_probs_up + (
            (1 - sigma_probs_down) * (1 - sigma_probs_up)), 0)
    return(sigma_probs_up, sigma_probs_down, prob_prime)


def compute_signal(
    sigma_probs_up, sigma_probs_down, prob_prime,sigma_probs_up_prev,
    sigma_probs_down_prev, prob_prime_prev, lower_threshold=15, n_signals=4):
    buy_signal_record = {"0": False}
    sell_signal_record = {"0": False}
    for signal_index in n_signals:
        buy_signal_record["%i" % signal_index] = False
        sell_signal_record["%i" % signal_index] = False

    lower_threshold_dec = lower_threshold / 100.0
    sell_using_prob_prime = prob_prime > lower_threshold_dec and prob_prime_prev == 0
    sell_base_signal = sigma_probs_up < 1 and sigma_probs_up_prev == 1
    buy_using_prob_prime = prob_prime == 0 and prob_prime_prev > lower_threshold_dec
    buy_base_signal = sigma_probs_down < 1 and sigma_probs_down_prev == 1
    buy_signal_record["0"] = buy_base_signal or buy_using_prob_prime
    sell_signal_record["0"] = sell_base_signal or sell_using_prob_prime
    sell_using_sigma_probs_up = [sell_base_signal]
    buy_using_sigma_probs_down = [buy_base_signal]
    if 1 in n_signals:
        signal_1_sell = sigma_probs_down_prev == 0 and sigma_probs_down > 0
        signal_1_buy = sigma_probs_up_prev == 0 and sigma_probs_up > 0
        buy_signal_record["%i" % 1] = signal_1_buy
        sell_signal_record["%i" % 1] = signal_1_sell
        sell_using_sigma_probs_up.append(signal_1_sell)
        buy_using_sigma_probs_down.append(signal_1_buy)
    if 2 in n_signals:
        signal_2_sell = sigma_probs_down_prev < 1 and sigma_probs_down == 1
        signal_2_buy = sigma_probs_up_prev > 0 and sigma_probs_up == 0
        buy_signal_record["%i" % 2] = signal_2_buy
        sell_signal_record["%i" % 2] = signal_2_sell
        sell_using_sigma_probs_up.append(signal_2_sell)
        buy_using_sigma_probs_down.append(signal_2_buy)
    buy_using_sigma_probs_down_cross = cross_over(
        [prob_prime_prev, prob_prime], [sigma_probs_down_prev, sigma_probs_down])
    sell_using_sigma_probs_down_cross = cross_under(
        [prob_prime_prev, prob_prime], [sigma_probs_down_prev, sigma_probs_down])
    if 3 in n_signals:
        signal_3_sell = sell_using_sigma_probs_down_cross and max(
            [prob_prime_prev, prob_prime]) > lower_threshold_dec
        signal_3_buy = buy_using_sigma_probs_down_cross and max(
            [prob_prime_prev, prob_prime]) > lower_threshold_dec
        buy_signal_record["%i" % 3] = signal_3_buy
        sell_signal_record["%i" % 3] = signal_3_sell
        sell_using_sigma_probs_up.append(signal_3_sell)
        buy_using_sigma_probs_down.append(signal_3_buy)
    buy_using_sigma_probs_up_cross = cross_over(
        [prob_prime_prev, prob_prime], [sigma_probs_up_prev, sigma_probs_up])
    sell_using_sigma_probs_up_cross = cross_under(
        [prob_prime_prev, prob_prime], [sigma_probs_up_prev, sigma_probs_up])
    if 4 in n_signals:
        signal_4_sell = False
        signal_4_buy = (
            sell_using_sigma_probs_up_cross and max(
                [prob_prime_prev, prob_prime]) > lower_threshold_dec) or (
                    buy_using_sigma_probs_up_cross and max(
                        [prob_prime_prev, prob_prime]) > lower_threshold_dec)
        buy_signal_record["%i" % 4] = signal_4_buy
        sell_signal_record["%i" % 4] = signal_4_sell
        # sell_using_sigma_probs_up.append(
        #     sell_using_sigma_probs_up_cross and max([prob_prime_prev, prob_prime]) > lower_threshold_dec)
        buy_using_sigma_probs_down.append(signal_4_buy)
    if 5 in n_signals:
        signal_5_sell = False
        signal_5_buy = sigma_probs_up > sigma_probs_down and sigma_probs_up > prob_prime and sigma_probs_up_prev > sigma_probs_up
        buy_signal_record["%i" % 5] = signal_5_buy
        sell_signal_record["%i" % 5] = signal_5_sell
        buy_using_sigma_probs_down.append(signal_5_buy)
        # sell_using_sigma_probs_up.append(
        #         sigma_probs_down_prev < 1 and sigma_probs_down == 1 and sigma_probs_down > sigma_probs_up and sigma_probs_down > prob_prime and sigma_probs_down)
    sell_signal = sell_using_prob_prime or any(sell_using_sigma_probs_up)
    buy_signal = buy_using_prob_prime or any(buy_using_sigma_probs_down)
    return (buy_signal, sell_signal, buy_signal_record, sell_signal_record)


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

def get_signal_from_dict(signal_id, signal_dict):
    try:
        signal = signal_dict['%i' % signal_id]
    except KeyError:
        signal = False
    return signal


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

def peaks_diff(peaks_values, min_diff=10):
    if len(peaks_values) > 2:
        first_peak = peaks_values[-1]
        second_peak = peaks_values[-2]
        p_diff = first_peak - second_peak
        if abs(p_diff) < min_diff:
            return 0
        else:
            return p_diff
    else:
        return 0
