"""
Ported from https://bango29.com/bayesian-probability-as-an-oscillator/
"""

from numpy import greater, less, sum, nan_to_num


def initialize(state):
    state.number_offset_trades = 0;
    state.bbres_last = {}
    state.orders = {}
    state.params = {}
    state.params["ZILUSDT"] = [0.12, 0.16, 15]
    state.params["MATICUSDT"] = [0.12, 0.1, 15]
    state.params["VITEUSDT"] = [0.12, 0.16, 15]


@schedule(interval="15m", symbol=["VITEUSDT", "MATICUSDT", "ZILUSDT"])
def handler(state, data):
    portfolio = query_portfolio()
    n_coins = len(data.keys())
    balance_quoted = portfolio.excess_liquidity_quoted
    # we invest only 80% of available liquidity
    #buy_value = float(balance_quoted) * percent_invest
    n_pos = 0
    for symbol_x in data.keys():
        position = query_open_position_by_symbol(
            symbol_x, include_dust=False)
        if position is not None:
           n_pos += 1
    amount = 0
    if n_pos < n_coins:
        amount = float(balance_quoted) / (n_coins - n_pos)
    for symbol_x in data.keys():
        handler_main(state, data[symbol_x], amount)

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


def handler_main(state, data, amount):
    symbol = data.symbol
    bb_period = 20
    try:
        params = state.params[symbol]
    except KeyError:
        params = [0.12, 0.16, 15]
    # stop_loss = 0.12
    # take_profit = 0.16
    # lower_threshold = 15

    stop_loss = params[0]
    take_profit = params[1]
    lower_threshold = params[2]

    stop_loss = 0.12
    take_profit = 0.16
    lower_threshold = 15

    bb_std_dev_mult = 2
    bbands = data.bbands(bb_period, bb_std_dev_mult)
    # lookback period
    bayes_period = 20

    percent_invest = 0.97
    
    # on erronous data return early (indicators are of NoneType)
    if bbands is None:
        return

    bbands_middle = bbands["bbands_middle"].last

    current_price = data.close_last
    bb_res = bbbayes(
        data.close.select('close'), bayes_period,
        bbands.select('bbands_upper'), bbands.select('bbands_lower'),
        bbands.select('bbands_middle'))

    plot("sigma_up", bb_res[0], symbol)
    plot("sigma_down", bb_res[1], symbol)
    plot("prime_prob", bb_res[2], symbol)
    #portfolio = query_portfolio()
    #balance_quoted = portfolio.excess_liquidity_quoted
    # we invest only 80% of available liquidity
    #buy_value = float(balance_quoted) * percent_invest
    buy_value = amount * percent_invest
    # buy_value = 100

    sigma_probs_up = bb_res[0]
    sigma_probs_down = bb_res[1]
    prob_prime = bb_res[2]
    try:
        sigma_probs_up_last = state.bbres_last[symbol][0]
    except KeyError:
        state.bbres_last[symbol] = [0, 0, 0]
        sigma_probs_up_last = 0
    sigma_probs_down_last = state.bbres_last[symbol][1]
    prob_prime_last = state.bbres_last[symbol][2]
    sell_using_prob_prime = prob_prime > lower_threshold / 100 and prob_prime_last == 0
    sell_using_sigma_probs_up = sigma_probs_up < 1 and sigma_probs_up_last == 1
    buy_using_prob_prime = prob_prime == 0 and prob_prime_last > lower_threshold / 100
    buy_using_sigma_probs_down = sigma_probs_down < 1 and sigma_probs_down_last == 1

    sell_signal = sell_using_prob_prime or sell_using_sigma_probs_up
    buy_signal = buy_using_prob_prime or buy_using_sigma_probs_down
    state.bbres_last[symbol] = bb_res
    buy_signal = buy_signal and bbands_middle > current_price

    position = query_open_position_by_symbol(
        data.symbol, include_dust=False)
    has_position = position is not None


    if buy_signal and not has_position:
        print("-------")
        print("Buy Signal: creating market order for {}".format(data.symbol))
        print("Buy value: ", buy_value, " at current market price: ", data.close_last)
        
        buy_order = order_market_value(symbol=data.symbol, value=buy_value)
        #order_stop_loss(
        #    symbol, buy_order.quantity,stop_loss,subtract_fees=True)
        make_double_barrier(
            symbol, float(buy_order.quantity), take_profit,
            stop_loss,state)

    elif sell_signal and has_position:
        print("-------")
        logmsg = "Sell Signal: closing {} position with exposure {} at current market price {}"
        print(logmsg.format(data.symbol,float(position.exposure),data.close_last))
        try:
            cancel_order(state.orders[symbol]['order_lower'].id)
            cancel_order(state.orders[symbol]['order_upper'].id)
        except KeyError:
            pass
        close_position(data.symbol)


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
        state.orders[symbol]
        state.orders[symbol]["order_upper"] = order_upper
        state.orders[symbol]["order_lower"] = order_lower
        state.orders[symbol]["created_time"] = order_upper.created_time
    except KeyError:
        state.orders[symbol] = {}
        state.orders[symbol]["order_upper"] = order_upper
        state.orders[symbol]["order_lower"] = order_lower
        state.orders[symbol]["created_time"] = order_upper.created_time


    return order_upper, order_lower
