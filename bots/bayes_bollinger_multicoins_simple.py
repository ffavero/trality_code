from numpy import greater, less, sum, nan_to_num


TITLE = "Multicoin Bollinger Bands Bayesian Oscillator"
VERSION = "24.1"
ALIAS = "PingPongGo"
AUTHOR = "Francesco @79bass 2021-04-28"
DONATE = ("TIP JAR WALLET:  \n" +  
           "ERC20:  0xEAC8a0d3AB6761860395b33f74dea88B4F16aBcA")


INTERVAL = "15m"
SYMBOLS = [
#    "ETHUSDT", "BTCUSDT", "ADAUSDT", "MATICUSDT", "RSRUSDT"]
    "VITEUSDT", "MATICUSDT", "ZILUSDT", "RUNEUSDT", "EGLDUSDT"]
VERBOSE = 1


## 0 prints only signals and portfolio information
## 1 prints updating orders info
## 2 prints stats info at each candle

SIGNALS = [1, 2]

"""
Disclaimer: This script came with no guarantee of making profits, if you sustain substantial
losses using this script I will take no responsability.
However if you are profiting from the bot you are welcome to tip me a coffee in the ERC20
address :) 

Changelog:
  Refactoring of the original bayesian bollinger bot
  - multiple coins trading
  - dynamic trailing market orders
  - tunable parameters for every coin
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
        - order_type: The type of order to use from trality, available options are "limit", "trailing" or
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
    state.signals = {}
    state.limit_orders = {}
    state.params = {}
    state.params["DEFAULT"] = {
        "stop_loss": 0.1,
        "take_profit": 0.16,
        "lower_threshold": 15,
        "bayes_period": 20}


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
@schedule(interval="30m", symbol="BNBUSDT")
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
    for symbol_x in state.signals:
        if state.signals[symbol_x] is not None:
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

    bb_period = 20
    bb_std_dev_mult = 2
    bbands = data.bbands(bb_period, bb_std_dev_mult)

    if bbands is None:
        return

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

    buy_signal, sell_signal = compute_signal(
        sigma_probs_up, sigma_probs_down, prob_prime, sigma_probs_up_prev,
        sigma_probs_down_prev, prob_prime_prev, lower_threshold, SIGNALS)

    state.bbres_prev[symbol] = bb_res

    if not trading_live:
        if VERBOSE >= 1:
            print("Skip first candle to gather signals")
        return

    position = query_open_position_by_symbol(
        data.symbol, include_dust=False)
    has_position = position is not None

    if amount <= 11 and not has_position:
        print("Skip %s, amount too low" % symbol)
        return
    try:
        stop_order = state.limit_orders[symbol]['order_lower']
    except KeyError:
        stop_order = None
    if has_position and stop_order is None:
        make_double_barrier(
            symbol, float(position.exposure),
            take_profit, stop_loss, state)

    if buy_signal and not has_position:
        signal_msg_data = {"symbol": symbol, "value": buy_value, "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Buy Signal: creating market order for %(symbol)s\n"
            "Buy value: %(value)s at current market price %(current_price)f\n"
            "++++++\n")
        buy_order = order_market_value(
            symbol=data.symbol, value=buy_value)
        make_double_barrier(
            symbol, float(buy_order.quantity), take_profit,
            stop_loss,state)
        print(signal_msg % signal_msg_data)

    elif sell_signal and has_position:
        signal_msg_data = {"symbol": symbol, "amount": position.exposure, "current_price": current_price}
        signal_msg = (
            "++++++\n"
            "Sell Signal: creating market order for %(symbol)s\n"
            "Sell amount: %(amount)s at current market price %(current_price)f\n"
            "++++++\n")
        close_position(symbol)
        cancel_state_limit_orders(state, symbol)
        print(signal_msg % signal_msg_data)


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
    lower_threshold_dec = lower_threshold / 100.0
    sell_using_prob_prime = prob_prime > lower_threshold_dec and prob_prime_prev == 0
    sell_using_sigma_probs_up = [
        sigma_probs_up < 1 and sigma_probs_up_prev == 1]
    buy_using_prob_prime = prob_prime == 0 and prob_prime_prev > lower_threshold_dec
    buy_using_sigma_probs_down = [
        sigma_probs_down < 1 and sigma_probs_down_prev == 1]
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
    sell_signal = sell_using_prob_prime or any(sell_using_sigma_probs_up)
    buy_signal = buy_using_prob_prime or any(buy_using_sigma_probs_down)
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
