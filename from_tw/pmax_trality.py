from numpy import sum, nan_to_num
from trality.indicator import linreg, ema

def initialize(state):
    state.number_offset_trades = 0;


@schedule(interval="1h", symbol="BTCUSDT")
def handler(state, data):
    '''
    1) Compute indicators from data
    '''
    atr_periods = 10
    mav_len = 10
    v_len = 9
    atr_multiplier = 3.0
    change_atr = True
    normalize_atr = True

    if change_atr:
        # something like a sma of trange 
        atr = data.tr().sma(atr_periods)
    else:
        atr = data.atr(atr_periods)
    valpha = 2 / float(mav_len + 1)

    # hl2 is maybe different, try with data.avgprice if it's an
    # equivalent of tradingview hl2
    src = data.avgprice().select('avgprice')
    vud1 = src[-v_len:] - src[-(v_len + 1):-1]
    vdd1 = src[-(v_len + 1):-1] - src[-v_len:]
    for i in range(v_len):
        if vud1[i] < 0:
            vud1[i] = 0
        if vdd1[i] < 0:
            vdd1[i] = 0
    vUD = sum(vud1)
    vDD = sum(vdd1)

    vCMO = nan_to_num((vUD-vDD)/(vUD+vDD))
    VAR = nan_to_num(
        valpha * abs(vCMO) * src) + ( 1 - valpha * abs(vCMO))
    VAR = VAR * nan_to_num(VAR[-2])
    wwalpha = 1 / mav_len
    WWMA = wwalpha * src + (1 - wwalpha)
    WWMA = WWMA * nan_to_num(WWMA[-2])

    if mav_len%2 == 0:
        zxLag = mav_len / 2
    else:
        zxLag = (mav_len - 1) / 2
    zxEMAData = (src + (src - src[int(zxLag)]))
    ZLEMA = ema(zxEMAData, mav_len)
    lrc = linreg(src, mav_len, 0)
    lrc1 = linreg(src,mav_len,1)
    lrs = (lrc-lrc1)
    TSF = linreg(src, mav_len, 0) + lrs
    # Just use ema, other mav can be implemented as needed
    # the original script had a param to select which mav to use....
    MAvg = data.ema(mav_len)
    # on erronous data return early (indicators are of NoneType)
    if atr is None or MAvg is None:
        return

    if normalize_atr:
        longStop = MAvg - (atr_multiplier * atr / data.close)
    else:
        longStop = MAvg - (atr_multiplier * atr)


    current_price = data.close_last
    
    '''
    2) Fetch portfolio
        > check liquidity (in quoted currency)
        > resolve buy value
    '''
    
    portfolio = query_portfolio()
    balance_quoted = portfolio.excess_liquidity_quoted
    # we invest only 80% of available liquidity
    buy_value = float(balance_quoted) * 0.80
    
    
    '''
    3) Fetch position for symbol
        > has open position
        > check exposure (in base currency)
    '''

    position = query_open_position_by_symbol(data.symbol,include_dust=False)
    has_position = position is not None

    '''
    4) Resolve buy or sell signals
        > create orders using the order api
        > print position information
        
    '''
    if True and not has_position:
        print("-------")
        print("Buy Signal: creating market order for {}".format(data.symbol))
        print("Buy value: ", buy_value, " at current market price: ", data.close_last)
        
        order_market_value(symbol=data.symbol, value=buy_value)

    elif True and has_position:
        print("-------")
        logmsg = "Sell Signal: closing {} position with exposure {} at current market price {}"
        print(logmsg.format(data.symbol,float(position.exposure),data.close_last))

        close_position(data.symbol)
       
    '''
    5) Check strategy profitability
        > print information profitability on every offsetting trade
    '''
    
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

