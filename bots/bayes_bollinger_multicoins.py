from numpy import greater, less, sum, nan_to_num
from datetime import datetime


TITLE = "Multicoin Bollinger Bands Bayesian Oscillator"
VERSION = "39.1"
ALIAS = "TestMac"
AUTHOR = "Francesco @79bass 2021-04-28"
DONATE = ("TIP JAR WALLET:  " +
          "BEP-20: 0xEAC8a0d3AB6761860395b33f74dea88B4F16aBcA"
          "ERC20:  0xEAC8a0d3AB6761860395b33f74dea88B4F16aBcA")

INTERVAL_MACD_COORDINATOR = "1h"
INTERVAL = "15m"
INTERVAL_BNB = "15m"

# INTERVAL_MACD_COORDINATOR = "6h"
# INTERVAL = "1h"
# INTERVAL_BNB = "1h"

SYMBOLS = [
   "VITEUSDT", "MATICUSDT", "RUNEUSDT", "ZILUSDT", "BTCDOWNUSDT"]

#   "ADAUSDT", "MATICUSDT", "ETHUSDT", "DOTUSDT", "SOLUSDT"]
#   "ZILUSDT", "MATICUSDT", "RUNEUSDT", "VITEUSDT", "BTCDOWNUSDT"]

VERBOSE = 1

## 0 prints only signals and portfolio information
## 1 prints updating orders info
## 2 prints stats info at each candle

SIGNALS = [1, 5]

"""
Disclaimer: This script came with no guarantee of making profits, if you sustain substantial
losses using this script I will take no responsability.
However if you are profiting from the bot you are welcome to tip me a coffee in the ERC20
address :) 

Changelog:
  Refactoring of the original bayesian bollinger bot
  - multiple coins trading
  - dynamic trailing market orders
  - tunable parameters for every coins
  - TODO: Portfolio weighted splits (eg you want to trade
   50% BTC and the remaining value to split it with other
   altcoins)

DOCS:

   The bot comes with default paramaters, but it is possible to change per-coin setting by
   adding a key to the `state.params` dictionary, referencing a specific symbol.
   The per-coin parameters can be partials, the missing options will be taken by the DEFAULT
   settings.

   A quick description of the available parameters is the following
   options:

        - stop_loss: The percentage [0-1] of maximum loss before dropping a trade. Default 0.12 (12%)
        - take_profit: The percentage [0-1] of maximum gain before taking profit. Default 0.16 (16%)
        - lower_threshold: The limit in the derivative probability to consider a trade with 
                           the prime_prob signal (other signal will not be affected by this). Default 15
        - bayes_period: The numbers of previous candles to compute the probabilities. Default 20
        - order_type: The type of order to use from trality, available options are "limit", "market", "trailing" or
                      "if_touched". Default "trailing"
        - limit_rate_candle: The percentage on the last candle to set the limit above (selling) or
                             below (buying), Default 0.75 (eg a sell; Last candle has 1% gains at price 100,
                             the limit will be set to 100.75)
"""



### Settings in state

def initialize(state):
    state.number_offset_trades = 0
    state.percent_invest = 0.98
    state.bbres_prev = {}
    state.limit_orders = {}
    state.signals = {}
    state.semaphore = {}
    state.zero_signal_timer = {}
    state.long_atr = {}
    state.fine_tuning = {}
    state.params = {}
    state.summary_performance = {}
    state.params["DEFAULT"] = {
        "atr_stop_loss_n": 6,
        "atr_take_profit_n": 6,
        "lower_threshold": 15,
        "bayes_period": 20,
        "order_type": "market",
        "limit_rate_candle": 0.75,
        "max_candles_with_0_signals": 24,
        "use_semaphore": True,
        "lenient_with_uptrend": True,
        "buy_with_ema45": False,
        "signals_mode": SIGNALS}
    state.params["VITEUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["ZILUSDT"] = {
        "signals_mode": [1, 3, 4, 5],
        "buy_with_ema45": False
    }
    state.params["RUNEUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["EGLDUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
### No fiddling from here below, all the settings are
### exposed in the state

# MACD coordinator
@schedule(
    interval=INTERVAL_MACD_COORDINATOR, symbol=SYMBOLS)
def coordinator_handler(state, data):
    try:
        for this_symbol in data.keys():
            coordinator_main(state, data[this_symbol])
    except TypeError:
        coordinator_main(state, data)



### Main handler
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
    for symbol_x in state.fine_tuning:
        if state.fine_tuning[symbol_x] is not None:
            n_pending += 1
    amount_bnb = 0
    if has_position:
        amount_bnb = position.position_value
    if buy_value >= add_bnb_amount and amount_bnb < low_bnb_amount:
        if n_pending == 0:
            print(
                "Buying BNB, current amount %f" % amount_bnb)
            order_market_value(data.symbol, value=add_bnb_amount)



### No fiddling from here below, all the settings are
### exposed in the state


### generic handler function with main strategy

def coordinator_main(state, data):
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

    position = query_open_position_by_symbol(
        symbol, include_dust=False)
    
    # has_position = position is not None

    macd_histogram_last = macd.select("macd_histogram")[-1]
    macd_histogram_2nd_last = macd.select("macd_histogram")[-2]
    this_semaphore = state.semaphore[symbol][1]
    #this_semaphore = "green"
    # if (macd_histogram_last < macd_histogram_2nd_last and macd_histogram_2nd_last < macd.select("macd_histogram")[-3]):
    #     this_semaphore = "red"
    # elif (macd_histogram_last > macd_histogram_2nd_last and macd_histogram_2nd_last > macd.select("macd_histogram")[-3]):
    #     this_semaphore = "green"
    if macd_histogram_last < 0 or current_price < vwma:
        if lenient_with_uptrend:
            if vma_diff < 0:
                this_semaphore = "red"
        else:
            this_semaphore = "red"
    elif macd_histogram_last > 0 or current_price > vwma:
        this_semaphore = "green"




    # if adx > atx_strong_trend_limit:
    #     state.params[symbol]["signals_mode"] = state.params[symbol]["signals_mode1"]
    # elif vma_diff > 0:
    #     state.params[symbol]["signals_mode"] = state.params[symbol]["signals_mode2"]
    # elif vma_diff < 0:
    #     state.params[symbol]["signals_mode"] = state.params[symbol]["signals_mode1"]

    # if adx >= 25:
    #     state.params[symbol]["signals_mode"] = [1]
    # elif adx < 25:
    #     state.params[symbol]["signals_mode"] = [1, 3, 4, 5]

    # if  adx >= 25 and vma_diff >= 0:
    #     state.params[symbol]["signals_mode"] = [1, 3, 4, 5]
    # elif adx >= 25 and vma_diff < 0:
    #     state.params[symbol]["signals_mode"] = [1]
    # elif adx < 25 and vma_diff < 0:
    #     state.params[symbol]["signals_mode"] = [1, 3, 4, 5]
    # else:
    #     state.params[symbol]["signals_mode"] = [1]


    # if macd_histogram_last < 0 or vma_diff < 0:
    #      this_semaphore = "red"
    # elif macd_histogram_last > 0 or vma_diff > 0:
    #      this_semaphore = "green"

    state.semaphore[symbol] = [last_semaphore, this_semaphore]


def handler_main(state, data, amount):
    if data is None:
        return
    symbol = data.symbol
    buy_value = amount
    params = get_default_params(state, symbol)

    stop_loss_n = params["atr_stop_loss_n"]
    take_profit_n = params["atr_take_profit_n"]
    # stop_loss = params["atr_stop_loss_n"]
    # take_profit = params["atr_take_profit_n"]    
    lower_threshold = params["lower_threshold"]
    bayes_period =  params["bayes_period"]
    order_type =  params["order_type"]
    limit_rate_candle = params["limit_rate_candle"]
    signals_mode =  params["signals_mode"]
    max_candles_with_0_signals = params["max_candles_with_0_signals"]
    use_semaphore = params["use_semaphore"]
    buy_with_ema45 = params["buy_with_ema45"]

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

    try:
        signal = state.signals[symbol]
    except KeyError:
        state.signals[symbol] = None
        signal = state.signals[symbol]


    trail_percent, trail_limit = dynamic_trailing_limits(
        data.open_last, data.close_last,
        limit_rate=limit_rate_candle, trailing_rate=limit_rate_candle)

    bb_period = 20
    bb_std_dev_mult = 2
    bbands = data.bbands(bb_period, bb_std_dev_mult)
    atr = data.atr(14).last
    adx_data = data.adx(14)
    fisher = data.fisher(14)
    wad = data.wad()
    adx = adx_data.last
    ema30 = data.ema(30).last
    cr45 = cross_over(data.ema(45).select("ema")[-2:], data.close.select("close")[-2:])
    cr45 = cr45 or cross_under(data.ema(45).select("ema")[-2:], data.close.select("close")[-2:])

    with PlotScope.group("ema45", symbol):
        plot("cross ema45", int(cr45))

    with PlotScope.group("wad", symbol):
        plot("wad", wad.last)
        plot("wad_sma", wad.sma(8).last)
        plot("wad_sma_long", wad.sma(20).last)

    adx_diff = adx - adx_data[-2]

    if bbands is None:
        return

    current_price = data.close_last
    stop_loss, sl_price = atr_tp_sl_percent(
        float(current_price), float(atr), stop_loss_n, False)
    take_profit, tp_price = atr_tp_sl_percent(
        float(current_price), float(atr), take_profit_n, True)


    bb_res = bbbayes(
        data.close.select('close'), bayes_period,
        bbands.select('bbands_upper'), bbands.select('bbands_lower'),
        bbands.select('bbands_middle'))

    last_two_close = data.close.select('close')[-2:]

    with PlotScope.group("bayesian_bollinger", symbol):
        plot("sigma_up", bb_res[0])
        plot("sigma_down", bb_res[1])
        plot("prime_prob", bb_res[2])


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

    if VERBOSE >= 2:
        print(
            "%(symbol)s:\n"
            "    sigma_probs_up: %(sigma_probs_up)f\n"
            "    sigma_probs_down: %(sigma_probs_up)f\n"
            "    prob_prime: %(prob_prime)f\n" % {
                "symbol": symbol,
                "sigma_probs_up": sigma_probs_up,
                "sigma_probs_down": sigma_probs_up,
                "prob_prime": prob_prime})

    buy_signal, sell_signal, a, b = compute_signal(
        sigma_probs_up, sigma_probs_down, prob_prime,
        sigma_probs_up_prev, sigma_probs_down_prev,
        prob_prime_prev, lower_threshold, signals_mode)
    if state.zero_signal_timer[symbol] >= max_candles_with_0_signals:
        sell_signal = True
    # if b["0"] == True and semaphores[1] == "red":
    #     semaphores[1] = "green"
    #     state.semaphore[symbol][1] = "green"
    #     state.semaphore_override[symbol] = True
    # if a["0"] == True and state.semaphore_override[symbol] == True:
    #     state.semaphore_override[symbol] = False

    # if b["0"] == True and buy_signal == True:
    #     buy_signal = False
    #     state.semaphore_override[symbol] = True
    # if a["0"] == True and sell_signal == True:
    #     sell_signal = False

    with PlotScope.group("semaphore", symbol):
        if semaphores[1] == "red":
            plot("semaphore", 0)
        elif semaphores[1] == "green":
            plot("semaphore", 1)

    with PlotScope.group("signal", symbol):
        plot("0", int(a["0"]) + (-1 * int(b["0"])))
    with PlotScope.group("signals", symbol):
        plot("buy", int(buy_signal))
        plot("sell", -1 * int(sell_signal))
    #     if len(a[1]) > 2:
    #         plot("3", int(a[1][1]) + (-1 * int(b[1][1])))
    #         plot("4", int(a[1][2]) + (-1 * int(b[1][2])))
    #         plot("5", int(a[1][3]) + (-1 * int(b[1][3])))

    state.bbres_prev[symbol] = bb_res

    if buy_with_ema45:
        buy_signal = cr45

    if buy_signal and ema30 > bbands.select('bbands_middle')[-1]:
        pass
    else:
        buy_signal = False

    if has_position and not sell_signal:
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


    if semaphores[1] == "red" and not has_position and use_semaphore:
        # if has_position:
        #     close_position(symbol)
        #     cancel_state_limit_orders(state, symbol)
        #     cancel_state_tuning_orders(state, symbol)
        #     state.signals[symbol] = None
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

    if signal is not None:
        if signal == "buy":
            try:
                buy_order = state.fine_tuning[symbol]
            except KeyError:
                raise(Exception("I can't find the fine tuning order"))
            buy_order.refresh()
            partially_filled = buy_order.status.value == 3
            order_status = [buy_order.is_filled(
                ), buy_order.is_error(
                    ), buy_order.is_rejected(
                        ), buy_order.is_canceled(), partially_filled]
            if any(order_status):
                if not any([buy_order.is_filled(), partially_filled]):
                # if filled reset signal, state and set stop/tp limits
                # If failed open market position
                    buy_order = order_market_value(symbol=symbol, value=buy_value)
                cancel_state_tuning_orders(state, symbol)
                state.signals[symbol] = None
                state.fine_tuning[symbol] = None
                make_double_barrier(symbol, float(buy_order.quantity), take_profit,
                    stop_loss, state)
            else:
                if has_position:
                    cancel_state_tuning_orders(state, symbol)
                    state.signals[symbol] = None
                else:
                    # update order
                    update_msg_data = {
                        "symbol": symbol, "value": buy_value, "current_price": current_price}
                    update_msg = (
                        "-------\n"
                        "Update order for %(symbol)s\n"
                        "Buy value: %(value)f at current market price: %(current_price)f")
                    if VERBOSE >= 1:
                        print(update_msg % update_msg_data)
                    update_or_init_buy_fine_tuning(
                        symbol, buy_value, last_two_close, trail_percent,
                        trail_limit, state, order_type)

        elif signal == "sell":

            # check if order is filled/failed
            try:
                sell_order = state.fine_tuning[symbol]
            except KeyError:
                raise("I can't find the fine tuning order")
            sell_order.refresh()
            partially_filled = sell_order.status.value == 3
            order_status = [sell_order.is_filled(
                ), sell_order.is_error(
                    ), sell_order.is_rejected(
                        ), sell_order.is_canceled(), partially_filled]
            if any(order_status):
                if not sell_order.is_filled():
                # if filled reset signal, state and set stop/tp limits
                # If failed open market position
                    close_position(symbol)
                cancel_state_tuning_orders(state, symbol)
                state.signals[symbol] = None
                state.fine_tuning[symbol] = None
            else:
                if not has_position:
                    cancel_state_limit_orders(state, symbol)
                    cancel_state_tuning_orders(state, symbol)
                    state.signals[symbol] = None
                else:
                    # update order
                    update_msg_data = {
                        "symbol": symbol, "amount": position.exposure, "current_price": current_price}
                    update_msg = (
                        "-------\n"
                        "Update sell order for %(symbol)s\n"
                        "sell amount: %(amount)f at current market price: %(current_price)f")
                    if VERBOSE >= 1:
                        print(update_msg % update_msg_data)
                    update_or_init_sell_fine_tuning(
                        symbol, position.exposure, last_two_close, trail_percent,
                        trail_limit, state, order_type)
    else:
        try:
            stop_order = state.limit_orders[symbol]['order_lower']
        except KeyError:
            stop_order = None
        if has_position and stop_order is None:
            make_double_barrier(
                symbol, float(position.exposure),
                take_profit, stop_loss, state)

    if buy_signal and not has_position and state.signals[symbol] is None:
        signal_msg_data = {"symbol": symbol, "value": buy_value, "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Buy Signal: creating market order for %(symbol)s\n"
            "Buy value: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        print(signal_msg % signal_msg_data)
        if order_type == "market":
            buy_order = order_market_value(
                symbol=symbol, value=buy_value)
            state.signals[symbol] = None
            state.fine_tuning[symbol] = None
            make_double_barrier(
                symbol, float(buy_order.quantity), take_profit,
                stop_loss, state)
        else:
            state.signals[symbol] = "buy"
            update_or_init_buy_fine_tuning(
                        symbol, buy_value, last_two_close,
                        trail_percent, trail_limit, state, order_type)

    elif sell_signal and has_position and state.signals[symbol] is None:
        state.signals[symbol] = "sell"
        signal_msg_data = {"symbol": symbol, "amount": position.exposure, "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Sell Signal: creating market order for %(symbol)s\n"
            "Sell amount: %(amount)s at current market price %(current_price)f\n"
            "++++++\n")
        print(signal_msg % signal_msg_data)
        cancel_state_limit_orders(state, symbol)
        if order_type == "market":
            state.signals[symbol] = None
            state.fine_tuning[symbol] = None
            close_position(symbol)
        else:
            update_or_init_sell_fine_tuning(
                symbol, float(position.exposure), last_two_close,
                trail_percent, trail_limit, state, order_type)



### methods and helpers

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

def cancel_state_tuning_orders(state, symbol):
    try:
        cancel_order(state.fine_tuning[symbol].id)
    except Exception:
        pass


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


def update_or_init_sell_fine_tuning(
    symbol, amount, current_prices, trailing_percent, stop_limit, state, order_type):
    going_down = current_prices[0] > current_prices[1]
    try:
        old_order = state.fine_tuning[symbol]
    except KeyError:
        old_order = None
    if going_down and old_order is not None:
        update_msg_data = {
            "symbol": symbol, "current_price": current_prices[1], "prev_price": current_prices[0]}
        update_msg = (
            "-------\n"
            "The current price  (%(current_price)f) is lower than the previous close: (%(prev_price)f)\n"
            "Skip update sell order for %(symbol)s\n")
        if VERBOSE >= 1:
            print(update_msg % update_msg_data)
        return
    # elif going_down and old_order is None:
    #     stop_limit = 0.001
    #     trailing_percent = 0.001

    current_price = current_prices[1]

    if old_order is not None:
        cancel_state_tuning_orders(state, symbol)

    stop_price = current_price * (1 + stop_limit)
    update_msg_data = {
        "symbol": symbol, "stop_price": stop_price}
    update_msg = (
        "-------\n"
        "Sell order for %(symbol)s with limit %(stop_price)f\n")
    if VERBOSE >= 1:
        print(update_msg % update_msg_data)
    if order_type == "trailing":
        tune_order = order_trailing_iftouched_amount(
            symbol, amount=-1 * float(amount), trailing_percent=trailing_percent,
            stop_price=stop_price)
    elif order_type == "if_touched":
        tune_order = order_iftouched_market_amount(
            symbol, amount=-1 * float(amount), stop_price=stop_price)
    elif order_type == "limit":
        tune_order = order_limit_amount(
            symbol, amount=-1 * subtract_order_fees(
                float(amount)), limit_price=stop_price)
    else:
        raise Exception("Supported orders type are only limit, trailing or if_touched")
    state.fine_tuning[symbol] = tune_order


def update_or_init_buy_fine_tuning(
    symbol, value, current_prices, trailing_percent, stop_limit, state, order_type):
    going_up = current_prices[0] < current_prices[1]
    try:
        old_order = state.fine_tuning[symbol]
    except KeyError:
        old_order = None
    if going_up and old_order is not None:
        update_msg_data = {
            "symbol": symbol, "current_price": current_prices[1], "prev_price": current_prices[0]}
        update_msg = (
            "-------\n"
            "The current price  (%(current_price)f) is higher than the previous close: (%(prev_price)f)\n"
            "Skip update buy order for %(symbol)s\n")
        if VERBOSE >= 1:
            print(update_msg % update_msg_data)
        return
    # elif going_up and old_order is None:
    #     stop_limit = 0.001
    #     trailing_percent = 0.001
    current_price = current_prices[1]

    if old_order is not None:
        cancel_state_tuning_orders(state, symbol)
        try:
            executed_value = old_order.executed_price * old_order.executed_quantity
        except TypeError:
            executed_value = 0

        value -= executed_value
    stop_price = current_price * (1 - stop_limit)
    update_msg_data = {
        "symbol": symbol, "stop_price": stop_price}
    update_msg = (
        "-------\n"
        "Buy order for %(symbol)s with limit %(stop_price)f\n")
    if VERBOSE >= 1:
        print(update_msg % update_msg_data)
    if order_type == "trailing":
        tune_order = order_trailing_iftouched_value(symbol, value=value, 
             trailing_percent=trailing_percent, stop_price=stop_price)
    elif order_type == "if_touched":
        tune_order = order_iftouched_market_value(
            symbol, value=value, stop_price=stop_price)
    elif order_type == "limit":
        tune_order = order_limit_value(
            symbol, value=value, limit_price=stop_price)
    else:
        raise Exception("Supported orders type are only limit, trailing or if_touched")
    state.fine_tuning[symbol] = tune_order


def dynamic_trailing_limits(open_price, close_price, limit_rate=0.75, trailing_rate=0.75):
    change_percent = abs((
        close_price - open_price) / open_price)
    stop_limit_percent = change_percent * limit_rate
    trailing_percent = change_percent * trailing_rate
    return (trailing_percent, stop_limit_percent)


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
        sell_using_sigma_probs_up.append(
            sigma_probs_down_prev == 0 and sigma_probs_down > 0)
        buy_using_sigma_probs_down.append(
            sigma_probs_up_prev == 0 and sigma_probs_up > 0)
    if 2 in n_signals:
        sell_using_sigma_probs_up.append(
            sigma_probs_down_prev < 1 and sigma_probs_down == 1)
        buy_using_sigma_probs_down.append(
            sigma_probs_up_prev > 0 and sigma_probs_up == 0)
    buy_using_sigma_probs_down_cross = cross_over(
        [prob_prime_prev, prob_prime], [sigma_probs_down_prev, sigma_probs_down])
    sell_using_sigma_probs_down_cross = cross_under(
        [prob_prime_prev, prob_prime], [sigma_probs_down_prev, sigma_probs_down])
    if 3 in n_signals:
        sell_using_sigma_probs_up.append(
            sell_using_sigma_probs_down_cross and max([prob_prime_prev, prob_prime]) > lower_threshold_dec)
        buy_using_sigma_probs_down.append(
            buy_using_sigma_probs_down_cross and max([prob_prime_prev, prob_prime]) > lower_threshold_dec)
    buy_using_sigma_probs_up_cross = cross_over(
        [prob_prime_prev, prob_prime], [sigma_probs_up_prev, sigma_probs_up])
    sell_using_sigma_probs_up_cross = cross_under(
        [prob_prime_prev, prob_prime], [sigma_probs_up_prev, sigma_probs_up])
    if 4 in n_signals:
        # sell_using_sigma_probs_up.append(
        #     sell_using_sigma_probs_up_cross and max([prob_prime_prev, prob_prime]) > lower_threshold_dec)
        buy_using_sigma_probs_down.append(
            sell_using_sigma_probs_up_cross and max([prob_prime_prev, prob_prime]) > lower_threshold_dec)
        buy_using_sigma_probs_down.append(
            buy_using_sigma_probs_up_cross and max([prob_prime_prev, prob_prime]) > lower_threshold_dec)
    if 5 in n_signals:
        buy_using_sigma_probs_down.append(
                sigma_probs_up > sigma_probs_down and sigma_probs_up > prob_prime and sigma_probs_up_prev > sigma_probs_up)
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

