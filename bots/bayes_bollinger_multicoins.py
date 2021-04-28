from numpy import greater, less, sum, nan_to_num


TITLE = "Multicoin Bollinger Bands Bayesian Oscillator"
VERSION = "21.1"
ALIAS = "Mjollnir"
AUTHOR = "Francesco @79bass 2021-04-28"
DONATE = ("TIP JAR WALLET:  \n" +  
           "ERC20:  0xc7F0A80f8a16F50067ABcd511f72a6D4eeAFC59c")


INTERVAL = "15m"
SYMBOLS = ["VITEUSDT", "MATICUSDT", "ZILUSDT", "RUNEUSDT", "EGLDUSDT"]

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
        - order_type: The type of order to use from trality, available options are "trailing" or
                      "if_touched". Default "if_touched"
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
    state.fine_tuning = {}
    state.params = {}
    state.params["DEFAULT"] = {
        "stop_loss": 0.12,
        "take_profit": 0.16,
        "lower_threshold": 15,
        "bayes_period": 20,
        "order_type": "if_touched",
        "limit_rate_candle": 0.75}


@schedule(interval=INTERVAL, symbol=SYMBOLS)

### No fiddling from here below, all the settings are
### exposed in the state

### Main handler

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
        print("-------")
        print("Accumulated Pnl of Strategy: {}".format(pnl))
        
        offset_trades = portfolio.number_of_offsetting_trades
        number_winners = portfolio.number_of_winning_trades
        print("Number of winning trades {}/{}.".format(number_winners,offset_trades))
        print("Best trade Return : {:.2%}".format(portfolio.best_trade_return))
        print("Worst trade Return : {:.2%}".format(portfolio.worst_trade_return))
        print("Average Profit per Winning Trade : {:.2f}".format(portfolio.average_profit_per_winning_trade))
        print("Average Loss per Losing Trade : {:.2f}".format(portfolio.average_loss_per_losing_trade))
        # reset number offset trades
        state.number_offset_trades = portfolio.number_of_offsetting_trades


### generic handler function with main strategy

def handler_main(state, data, amount):
    symbol = data.symbol

    params = get_default_params(state, symbol)

    try:
        signal = state.signals[symbol]
    except KeyError:
        state.signals[symbol] = None
        signal = state.signals[symbol]

    stop_loss = params["stop_loss"]
    take_profit = params["take_profit"]
    lower_threshold = params["lower_threshold"]
    bayes_period =  params["bayes_period"]
    order_type =  params["order_type"]
    limit_rate_candle = params["limit_rate_candle"]

    trail_percent, trail_limit = dynamic_trailing_limits(
        data.open_last, data.close_last,
        limit_rate=limit_rate_candle, trailing_rate=limit_rate_candle)

    bb_period = 20
    bb_std_dev_mult = 2
    bbands = data.bbands(bb_period, bb_std_dev_mult)
    if bbands is None:
        return
    bbands_middle = bbands["bbands_middle"].last

    current_price = data.close_last
    bb_res = bbbayes(
        data.close.select('close'), bayes_period,
        bbands.select('bbands_upper'), bbands.select('bbands_lower'),
        bbands.select('bbands_middle'))
    last_two_close = data.close.select('close')[-2:]

    with PlotScope.group("bayesian_bollinger", symbol):
        plot("sigma_up", bb_res[0])
        plot("sigma_down", bb_res[1])
        plot("prime_prob", bb_res[2])

    buy_value = amount

    sigma_probs_up = bb_res[0]
    sigma_probs_down = bb_res[1]
    prob_prime = bb_res[2]
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

    buy_signal, sell_signal = compute_signal(
        sigma_probs_up, sigma_probs_down, prob_prime, sigma_probs_up_prev,
        sigma_probs_down_prev, prob_prime_prev, bbands_middle, current_price, lower_threshold)

    state.bbres_prev[symbol] = bb_res

    if not trading_live:
        print("Skip first candle to gather signals")
        return

    position = query_open_position_by_symbol(
        data.symbol, include_dust=False)
    has_position = position is not None

    if signal is not None:
        if signal == "buy":
            try:
                buy_order = state.fine_tuning[symbol]
            except KeyError:
                raise(Exception("I can't find the fine tuning order"))
            buy_order.refresh()
            order_status = [buy_order.is_filled(
                ), buy_order.is_error(
                    ), buy_order.is_rejected(
                        ), buy_order.is_canceled()]
            if any(order_status):
                if not buy_order.is_filled():
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
                    return
                # update order
                print("-------")
                print("Update order for {}".format(data.symbol))
                print("Buy value: ", buy_value, " at current market price: ", current_price)
                update_or_init_buy_fine_tuning(
                    symbol, buy_value, last_two_close, trail_percent, trail_limit, state, order_type)

        elif signal == "sell":

            # check if order is filled/failed
            try:
                sell_order = state.fine_tuning[symbol]
            except KeyError:
                raise("I can't find the fine tuning order")
            sell_order.refresh()
            order_status = [sell_order.is_filled(
                ), sell_order.is_error(
                    ), sell_order.is_rejected(
                        ), sell_order.is_canceled()]
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
                    return
                # update order
                print("Update sell position for {}".format(data.symbol))
                update_or_init_sell_fine_tuning(
                    symbol, sell_order.quantity, last_two_close, trail_percent,
                    trail_limit, state, order_type)
            pass
        return
    else:
        try:
            stop_order = state.limit_orders[symbol]['order_lower']
        except KeyError:
            stop_order = None
        if has_position and stop_order is None:
            make_double_barrier(
                symbol, float(position.exposure),
                take_profit, stop_loss, state)

    if buy_signal and not has_position:
        print("-------")
        print("Buy Signal: creating market order for {}".format(data.symbol))
        print("Buy value: ", buy_value, " at current market price: ", current_price)
        state.signals[symbol] = "buy"
        update_or_init_buy_fine_tuning(
                    symbol, buy_value, last_two_close, trail_percent, trail_limit, state, order_type)

    elif sell_signal and has_position:
        state.signals[symbol] = "sell"
        print("-------")
        logmsg = "Sell Signal: closing {} position with exposure {} at current market price {}"
        print(logmsg.format(data.symbol,float(position.exposure),current_price))
        cancel_state_limit_orders(state, symbol)
        update_or_init_sell_fine_tuning(
            symbol, float(position.exposure), last_two_close, trail_percent,
            trail_limit, state, order_type)



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
    try:
        old_order = state.fine_tuning[symbol]
    except KeyError:
        old_order = None
    if current_prices[0] > current_prices[1] and old_order is not None:
        return
    current_price = current_prices[1]

    if old_order is not None:
        amount = old_order.quantity
        cancel_state_tuning_orders(state, symbol)
    stop_price = current_price * (1 + stop_limit)

    if order_type == "trailing":
        tune_order = order_trailing_iftouched_amount(
            symbol, amount=-1 * float(amount), trailing_percent=trailing_percent,
            stop_price=stop_price)
    elif order_type == "if_touched":
        tune_order = order_iftouched_market_amount(
            symbol, amount=-1 * float(amount), stop_price=stop_price)
    else:
        raise Exception("Supported orders type are only trailing or if_touched")
    state.fine_tuning[symbol] = tune_order


def update_or_init_buy_fine_tuning(
    symbol, value, current_prices, trailing_percent, stop_limit, state, order_type):
    try:
        old_order = state.fine_tuning[symbol]
    except KeyError:
        old_order = None
    if current_prices[0] < current_prices[1] and old_order is not None:
        return

    current_price = current_prices[1]

    if old_order is not None:
        cancel_state_tuning_orders(state, symbol)
    stop_price = current_price * (1 - stop_limit)
    if order_type == "trailing":
        tune_order = order_trailing_iftouched_value(symbol, value=value, 
             trailing_percent=trailing_percent, stop_price=stop_price)
    elif order_type == "if_touched":
        tune_order = order_iftouched_market_value(
            symbol, value=value, stop_price=stop_price)
    else:
        raise Exception("Supported orders type are only trailing or if_touched")
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
                1 - prob_up_sma)))
    # Next candles are breaking Up
    prob_down_bb_upper = prob_bb_upper_down / (
        prob_bb_upper_down + prob_bb_upper_up)
    prob_down_bb_basis = prob_bb_basis_down / (
        prob_bb_basis_down + prob_bb_basis_up)
    prob_down_sma = prob_sma_down / (prob_sma_down + prob_sma_up)
    sigma_probs_up = nan_to_num(
        prob_down_bb_upper * prob_down_bb_basis * prob_down_sma / prob_down_bb_upper * prob_down_bb_basis * prob_down_sma + (
            (1 - prob_down_bb_upper) * (1 - prob_down_bb_basis) * (1 - prob_down_sma) ))

    prob_prime = nan_to_num(
        sigma_probs_down * sigma_probs_up / sigma_probs_down * sigma_probs_up + (
            (1 - sigma_probs_down) * (1 - sigma_probs_up)))
    return(sigma_probs_up, sigma_probs_down, prob_prime)


def compute_signal(
  sigma_probs_up, sigma_probs_down, prob_prime,sigma_probs_up_prev,
  sigma_probs_down_prev, prob_prime_prev,bbands_middle, current_price, lower_threshold=15):
    sell_using_prob_prime = prob_prime > lower_threshold / 100 and prob_prime_prev == 0
    sell_using_sigma_probs_up = (sigma_probs_up < 1 and sigma_probs_up_prev == 1) or (
        sigma_probs_down_prev == 0 and sigma_probs_down > 0)
    buy_using_prob_prime = prob_prime == 0 and prob_prime_prev > lower_threshold / 100
    buy_using_sigma_probs_down = (sigma_probs_down < 1 and sigma_probs_down_prev == 1) or (
        sigma_probs_up_prev == 0 and sigma_probs_up > 0)
    buy_using_sigma_probs_down_cross = cross_over(
        [prob_prime_prev, prob_prime], [sigma_probs_down_prev, sigma_probs_down])
    sell_using_sigma_probs_down_cross = cross_under(
        [prob_prime_prev, prob_prime], [sigma_probs_down_prev, sigma_probs_down])
    buy_using_sigma_probs_up_cross = cross_under(
        [prob_prime_prev, prob_prime], [sigma_probs_up_prev, sigma_probs_up])
    sell_using_sigma_probs_up_cross = cross_over(
        [prob_prime_prev, prob_prime], [sigma_probs_up_prev, sigma_probs_up])
    sell_signal = sell_using_prob_prime or sell_using_sigma_probs_up or sell_using_sigma_probs_down_cross #or sell_using_sigma_probs_up_cross
    buy_signal = buy_using_prob_prime or buy_using_sigma_probs_down or buy_using_sigma_probs_down_cross #or sell_using_sigma_probs_up_cross
    buy_signal = buy_signal #and bbands_middle > current_price
    return (buy_signal, sell_signal)


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
    return params
