##+------------------------------------------------------------------+
##| BAYESIAN BBANDS | 15m                                             |
##+------------------------------------------------------------------+

SYMBOLS_1 = "EGLDUSDT"
SYMBOLS1 = ["VITEUSDT", "MATICUSDT", "RUNEUSDT", "ZILUSDT", "1INCHUSDT"]
SYMBOLS3 = ["LUNAUSDT", "COCOSUSDT", "NKNUSDT", "NEOUSDT", "NANOUSDT"]
#SYMBOLS2 = ["MIRUSDT", "ZRXUSDT", "MANAUSDT", "CLVUSDT", "ALGOUSDT", "BNBUSDT"]
SYMBOLS2 = ["MANAUSDT", "IRISUSDT", "BNBUSDT"]


### TODO
# whitelist signal within the cooldown regions
# Classify regions to predict possibly losses
# Fix stop loss updating to a lower value (test if fix is necessary)


# Testing pairs 15-05/25-08 to evaluate signal 4
# with signal 4
# 24.08.2021-22:00:00> MATICUSDT winning positions 61/96, realized pnl: 1563.925
# 24.08.2021-22:00:00> VITEUSDT winning positions 71/109, realized pnl: -54.147
# 24.08.2021-22:00:00> 1INCHUSDT winning positions 57/95, realized pnl: 647.681
# 24.08.2021-22:00:00> NKNUSDT winning positions 66/109, realized pnl: 912.231
# 24.08.2021-22:00:00> NEOUSDT winning positions 65/103, realized pnl: 724.964
# 24.08.2021-22:00:00> NANOUSDT winning positions 50/96, realized pnl: -88.331
# 24.08.2021-22:00:00> RUNEUSDT winning positions 56/93, realized pnl: 1971.343
# 24.08.2021-22:00:00> ZILUSDT winning positions 68/112, realized pnl: 735.102
# 24.08.2021-22:00:00> COCOSUSDT winning positions 53/88, realized pnl: 1211.550
# 24.08.2021-22:00:00> LUNAUSDT winning positions 56/106, realized pnl: 522.449
# 24.08.2021-22:00:00> IRISUSDT winning positions 66/106, realized pnl: 1001.146

# no sign 4
# 24.08.2021-22:00:00> MATICUSDT winning positions 39/62, realized pnl: 1179.623
# 24.08.2021-22:00:00> VITEUSDT winning positions 54/82, realized pnl: -90.757
# 24.08.2021-22:00:00> 1INCHUSDT winning positions 43/63, realized pnl: 533.549
# 24.08.2021-22:00:00> NKNUSDT winning positions 35/60, realized pnl: 326.310
# 24.08.2021-22:00:00> NEOUSDT winning positions 48/75, realized pnl: 320.360
# signals 3 and 4
# 24.08.2021-22:00:00> RUNEUSDT winning positions 61/120, realized pnl: 2446.344
# 24.08.2021-22:00:00> VITEUSDT winning positions 75/144, realized pnl: -40.648
# 24.08.2021-22:00:00> ZILUSDT winning positions 73/143, realized pnl: 501.257
# 24.08.2021-22:00:00> COCOSUSDT winning positions 64/130, realized pnl: 1076.185
# 24.08.2021-22:00:00> LUNAUSDT winning positions 59/144, realized pnl: 241.610


INTERVAL = "15m"

N_SYMBOLS = 12                  # Define how many symbols we are trading
                                # in all handlers combined

LEVERAGE =  2                   # Multiply the amount for a given number
                                # eg balance is 1000 usdt, N_SYMBOLS is 10
                                # LEVERAGE is 2; instead of 100 usdt per trade
                                # the bot will use 200 usdt (if enough balance
                                # is available)

FIX_BUY_AMOUNT = None           # Specify a fix amount for trade if needed
                                # This will override any preference set by the
                                # N_SYMBOLS and LEVERAGE options


##+------------------------------------------------------------------+
##| SELL Options                                                     |
##+------------------------------------------------------------------+

ATR_TAKE_PROFIT = 6	            # A multiplier on the ATR value (e.g. 4)
ATR_STOP_LOSS = 6               # A multiplier on the ATR value (e.g. 6)
COLLECT_DATA = False            # if True a python dictionary with the trade data
                                # is printed at the end of each day
SIGNALS = [1, 4, 5]                # Signal to include, possible number 1, 2, 3, 4, 5
                                # Suggested values [1, 5]. For high volatile symbols [1, 3, 4, 5]


import numpy as np
from numpy import greater, less, sum, nan_to_num, exp
from datetime import datetime
from trality.indicator import mfi, ema

##+------------------------------------------------------------------+
##| Settings in state (could set different tp/sl for each symbol)    |
##+------------------------------------------------------------------+

def initialize(state):
    state.number_offset_trades = 0
    state.zero_signal_timer = {}
    state.params = {}
    state.past_daily_candles = {}
    state.hourly_candles = {}
    state.cooldown = {}
    state.balance_quoted = 0
    state.params["DEFAULT"] = {
        "bolligner_period": 20,
        "bolligner_sd": 2,
        "keltner_ema": 20,
        "keltner_atr": 20,
        "keltner_n": 2,
        "ema_longer_period": 50,
        "ema_long_period": 40,
        "ema_short_period": 10,
        "atr_stop_loss": ATR_STOP_LOSS,
        "atr_take_profit": ATR_TAKE_PROFIT,
        "max_loss_percent": None,
        "lower_threshold": 15,
        "bayes_period": 20,
        "keltner_filter": True,
        "ema_filter": True,
        "multistrategy": True,
        "keep_signal": 10,
        "use_cooldown": True,
        "max_candels_with_0_signals": 24,
        "signals_mode": SIGNALS,
        "collect_data": COLLECT_DATA}
    state.params["RUNEUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["ZILUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["EGLDUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["COCOSUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["LUNAUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["MIRUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["CLVUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }
    state.params["VITEUSDT"] = {
        "signals_mode": [1, 3, 4, 5]
    }

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
    if state.collect_data is not None:
        if int(datetime.fromtimestamp(data[list(data.keys())[0]].last_time / 1000).minute) == 0:
            if int(datetime.fromtimestamp(data[list(data.keys())[0]].last_time / 1000).hour) == 0:
                print(state.collect_data)


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


def signal_no_wait(position_manager, trade_data, indicators_data):
    position_manager.stop_waiting()
    return True


def signal_buy_cooldown(position_manager, trade_data, indicators_data):
    signal = False
    cooldown = indicators_data["cooldown"]
    if cooldown == False:
        position_manager.stop_waiting()
        signal = True
    return signal

def signal_sell_cci(position_manager, trade_data, indicators_data):
    signal = False
    adx = indicators_data["adx"]["data"]
    cci = indicators_data["cci"]["data"]
    if adx[-1] < 25 or cci[-1] < 100:
        signal = True
        position_manager.stop_waiting()
    return signal

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
    atr_stop_loss = params["atr_stop_loss"]
    atr_take_profit = params["atr_take_profit"]
    collect_data = params["collect_data"]
    bolligner_period = params["bolligner_period"]
    bolligner_sd = params["bolligner_sd"]
    keltner_ema = params["keltner_ema"]
    keltner_atr = params["keltner_atr"]
    keltner_n = params["keltner_n"]

    ema_longer_period = params["ema_longer_period"]
    ema_long_period = params["ema_long_period"]
    ema_short_period = params["ema_short_period"]

    lower_threshold = params["lower_threshold"]
    bayes_period =  params["bayes_period"]
    signals_mode =  params["signals_mode"]
    max_candels_with_0_signals = params["max_candels_with_0_signals"]
    max_loss_percent = params["max_loss_percent"]

    keltner_filter = params["keltner_filter"]
    ema_filter = params["ema_filter"]
    multistrategy = params["multistrategy"]
    keep_signal = params["keep_signal"]
    use_cooldown = params["use_cooldown"]
    lift_sl = True

    try:
        zero_signal_timer = state.zero_signal_timer[symbol]
    except KeyError:
        state.zero_signal_timer[symbol] = 0
        zero_signal_timer = state.zero_signal_timer[symbol]

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

    try:
        cooldown = state.cooldown[symbol]
    except KeyError:
        state.cooldown[symbol] = False
        cooldown = state.cooldown[symbol]
    last_cooldown = cooldown

    #------------#
    # Indicators #
    #------------#


    cci_data = data.cci(20)
    cci = cci_data.last
    adx_data = data.adx(14)
    adx = adx_data.last
    atr_data = data.atr(12)
    atr = atr_data.last
    engulfing = data.cdlengulfing().last

    ema_long_data = data.ema(ema_long_period)
    ema_short_data = data.ema(ema_short_period)
    ema_long = ema_long_data.last
    ema_short = ema_short_data.last

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

    if before_yesterday_candle:
        pp2 = (before_yesterday_candle["high"] +before_yesterday_candle["low"] + before_yesterday_candle["close"]) / 3
        past_r1 = 2 * pp2 - before_yesterday_candle["low"]
    else:
        past_r1 = None

    take_last = 70
    max_1h_candles = 500

    bbands = data.bbands(
        bolligner_period, bolligner_sd)
    kbands = keltner_channels(
        data, keltner_ema, keltner_atr, keltner_n, take_last)

    current_price = data.close_last
    current_low = data.low_last
    mid_low_point = 0.995 * (
        bbands.select("bbands_middle")[-1] + bbands.select("bbands_lower")[-1]) / 2

    if len(hourly_candles) == 0:
        
        new_hourly_candles = list(
            get_1h_candles(
                data.times, data.open, data.close,
                data.high, data.low, data.volume)
        )
    else:
        new_hourly_candles = list(
            get_1h_candles(
                data.times[-4:], data.select("open")[-4:],
                data.select("close")[-4:], data.select("high")[-4:],
                data.select("low")[-4:], data.select("volume")[-4:])
        )
    if len(new_hourly_candles) > 0:
        hourly_candles += new_hourly_candles

    if len(hourly_candles) > max_1h_candles:
        hourly_candles = hourly_candles[-max_1h_candles:]
    state.hourly_candles[symbol] = hourly_candles

    h_close = [ochl["close"] for ochl in hourly_candles]
    h_high = [ochl["high"] for ochl in hourly_candles]
    h_low = [ochl["low"] for ochl in hourly_candles]
    h_volume = [ochl["volume"] for ochl in hourly_candles]

    h_hlcv_array = np.asarray((
        h_high, h_low, h_close, h_volume), dtype=np.float32)

    mfi_1h_period = 48
    if len(h_close) >  mfi_1h_period * 2:
        mfi_1h = mfi(h_hlcv_array, mfi_1h_period)[0]
    else:
        mfi_1h = [None]

    min_mfi = 30


    last_closes = data.close.select("close")[-take_last:]
    last_ccis = cci_data.select("cci")[-take_last:]
    last_adxs = adx_data.select("dx")[-take_last:]

    bbands_above_keltner_up = bbands.select(
        'bbands_upper')[-1] > kbands['high'][-1]
    bbands_below_keltner_low = bbands.select(
        'bbands_lower')[-1] < kbands['low'][-1] 


    if past_r1 and past_r1 < r1 and current_price > kbands["high"][-1] and bbands_above_keltner_up:
        state.cooldown[symbol] = True
    elif current_price > r1 and current_price > kbands["high"][-1] and bbands_above_keltner_up:
        state.cooldown[symbol] = True
    # elif current_price < s1 and not below_1h_ema_longer:
    #     state.cooldown[symbol] = True


    if data.close.select("close")[-2] < kbands["low"][-2] and current_price > kbands["low"][-1] and bbands_below_keltner_low:
        state.cooldown[symbol] = False
    
    cooldown =  state.cooldown[symbol]
    if not use_cooldown:
        cooldown = False



    #------------------------------#
    # Derivates, Peaks and Valleys #
    #------------------------------#

    """
    This section is not useful yet, possibly use of this to 
    cleanup/confirm signals
    """



    # last_ccis_peaks = detect_peaks(
    #     last_ccis, mpd=8, edge=None, kpsh=True)
    # last_ccis_valleys = detect_peaks(
    #     last_ccis, mpd=8, edge=None, valley=True, kpsh=True)  

    # last_adxs_peaks = detect_peaks(
    #     last_adxs, mpd=8, edge=None, kpsh=True)
    # last_adxs_valleys = detect_peaks(
    #     last_adxs, mpd=8, edge=None, valley=True, kpsh=True)

    # last_valleys = find_local_min(last_closes)
    # last_peaks = find_local_max(last_closes)
    # last_ccis_valleys = find_local_min(last_ccis)
    # last_ccis_peaks = find_local_max(last_ccis)
    # last_adxs_valleys = find_local_min(last_adxs)
    # last_adxs_peaks = find_local_max(last_adxs)

    # last_close_valleys_values = last_closes[last_valleys]
    # last_cci_valleys_values = last_ccis[last_ccis_valleys]
    # last_adx_valleys_values = last_adxs[last_adxs_valleys]

    # last_close_peaks_values = last_closes[last_peaks]
    # last_cci_peaks_values = last_ccis[last_ccis_peaks]
    # last_adx_peaks_values = last_adxs[last_adxs_peaks]


    """
    Classify Market condition (TODO)

    """
    # volatility_state = [-1, 0, 1]
    # trend_state = [-1, 0 ,1]
    # def get_market_state(vol, trend):
    #     if vol == -1 and trend == -1:
    #         return 0
    #     elif vol == -1 and trend == 0:
    #         return 1
    #     elif vol == -1 and trend == 1:
    #         return 2
    #     elif vol == 0 and trend == -1:
    #         return 3
    #     elif vol == 0 and trend == 0:
    #         return 4
    #     elif vol == 0 and trend == 1:
    #         return 5
    #     elif vol == 1 and trend == -1:
    #         return 6
    #     elif vol == 1 and trend == 0:
    #         return 7
    #     elif vol == 1 and trend == 1:
    #         return 8

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
        "cross": {
            "long": ema_long_data.select("ema").tolist()[-5:],
            "short": ema_short_data.select("ema").tolist()[-5:]
        },
        "engulfing": int(engulfing),
        "bollinger": {
            "upper": float(bbands.select("bbands_upper")[-1]), 
            "middle":float(bbands.select("bbands_middle")[-1]), 
            "lower": float(bbands.select("bbands_lower")[-1])
        },
        "keltner": {
            "upper": float(kbands["high"][-1]), 
            "middle":float(kbands["middle"][-1]), 
            "lower": float(kbands["low"][-1])
        },
        "cooldown": cooldown
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



    if atr_stop_loss is not None:
        stop_loss, sl_price = atr_tp_sl_percent(
            float(current_price), float(atr), atr_stop_loss, False)
        if max_loss_percent is not None:
            if stop_loss > max_loss_percent:
                stop_loss = max_loss_percent
    if atr_take_profit is not None:
        take_profit, tp_price = atr_tp_sl_percent(
            float(current_price), float(atr), atr_take_profit, True)


    """Place stop loss for manually added positions"""
    if position_manager.has_position and not position_manager.is_stop_placed():
        position_manager.double_barrier(take_profit, stop_loss)


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
    Lift the stop loss at the mid-bollinger if the sl is lower than
    the entry price and the current price passed the middle bband
    """

    # if entry_price and sl_price:
    #     if sl_price < entry_price:
    #         if current_low > bbands.select("bbands_middle")[-1]:
    #             position_manager.update_double_barrier(
    #                 current_price,
    #                 stop_loss=price_to_percent(
    #                     current_price, bbands.select("bbands_middle")[-1]))

    """
    If position and the current price is above the mid bollinger
    keep updating the sl to the mid-bollinge
    """

    if lift_sl and position_manager.has_position and sl_price:
        if sl_price < bbands.select("bbands_middle")[-1]:
            if bbands_below_keltner_low and (
                current_low > bbands.select("bbands_middle")[-1]):
                position_manager.update_double_barrier(
                    current_price,
                    stop_loss=price_to_percent(
                        current_price, mid_low_point))

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


    #----------------------------------------------------#
    # Bayesian Bollinger compute probability indicators  #
    #----------------------------------------------------#
    
    bb_res = bbbayes(
        data.close.select('close'), bayes_period,
        bbands.select('bbands_upper'), bbands.select('bbands_lower'),
        bbands.select('bbands_middle'))

    """
    Compute the probability for the previous candle
    """

    bb_res_prev = bbbayes(
        data.close.select('close')[:-2], bayes_period,
        bbands.select('bbands_upper')[:-2], bbands.select('bbands_lower')[:-2],
        bbands.select('bbands_middle')[:-2])

    sigma_probs_up = bb_res[0]
    sigma_probs_down = bb_res[1]
    prob_prime = bb_res[2]

    if (sigma_probs_up + sigma_probs_down + prob_prime) == 0:
        state.zero_signal_timer[symbol] += 1
    else:
        state.zero_signal_timer[symbol] = 0

    sigma_probs_up_prev = bb_res_prev[0]
    sigma_probs_down_prev = bb_res_prev[1]
    prob_prime_prev = bb_res_prev[2]

    buy_signal_wait, sell_signal_wait, a, b = compute_signal(
        sigma_probs_up, sigma_probs_down, prob_prime,
        sigma_probs_up_prev, sigma_probs_down_prev,
        prob_prime_prev, lower_threshold, signals_mode)


    #----------------#
    # Resolve signal #
    #----------------#

    buy_signal = False
    sell_signal = False


    # resolve sell signals
    sell_0 = get_signal_from_dict(0, b)
    sell_1 = get_signal_from_dict(1, b)
    sell_2 = get_signal_from_dict(2, b)
    sell_3 = get_signal_from_dict(3, b)
    sell_4 = get_signal_from_dict(4, b)
    sell_5 = get_signal_from_dict(5, b)
    # resolve buy signals
    buy_0 = get_signal_from_dict(0, a)
    buy_1 = get_signal_from_dict(1, a)
    buy_2 = get_signal_from_dict(2, a)
    buy_3 = get_signal_from_dict(3, a)
    buy_4 = get_signal_from_dict(4, a)
    buy_5 = get_signal_from_dict(5, a)
    


    ema_filter_override = False

    default_trade_data = {
        "signal_type": None,
        "status": None,
        "n_active": 0,
        "level": 0
    }


    """
    Filter using keltner channels
    """
    if keltner_filter:
        if bbands_above_keltner_up and bbands_below_keltner_low:
            buy_signal_wait = False

    """
    Skip the ema filter if the keltner mid line is
    above the bollinger mid line
    """
    if bbands.select("bbands_middle")[-1] > kbands["middle"][-1]:
        if not bbands_above_keltner_up and not bbands_below_keltner_low:
            ema_filter_override = True        

    """
    Filter with ema
    """
    if not ema_filter_override and ema_filter:
        if buy_signal_wait and ema_long > ema_short:
            buy_signal_wait = False
  
    """
    Filter buy orders when long mfi gives too much
    oversold signal
    """  
    if mfi_1h[-1]:
        if mfi_1h[-1] < min_mfi:
            buy_signal_wait = False

    """
    Filter sell orders when break up
    """
    if bbands_above_keltner_up:
        if ema_long < ema_short:
            sell_signal_wait = False
    """
    Cancel out a buy signal if a sell signal is also
    fired in the same candle
    """
    if sell_signal_wait:
        buy_signal_wait = False

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


    if multistrategy:
        """
        Reset the trade data when a new signal pops up
        """
        if  not position_manager.has_position and buy_signal_wait :
            trade_data = default_trade_data
            trade_data["status"] = "buying"
            if buy_0:
                trade_data["signal_type"] = 0
            elif buy_1:
                trade_data["signal_type"] = 1
            elif buy_5:
                trade_data["signal_type"] = 5
            elif buy_3:
                trade_data["signal_type"] = 3
            elif buy_4:
                trade_data["signal_type"] = 4
            position_manager.start_waiting(trade_data, "waiting to buy")
        elif position_manager.has_position and sell_signal_wait:
            trade_data = default_trade_data
            trade_data["status"] = "selling"
            if sell_0:
                trade_data["signal_type"] = 0
            elif sell_1:
                trade_data["signal_type"] = 1
            elif sell_5:
                trade_data["signal_type"] = 5
            elif sell_3:
                trade_data["signal_type"] = 3
            elif sell_4:
                trade_data["signal_type"] = 4
            position_manager.start_waiting(trade_data, "waiting to sell")
    else:
        buy_signal = buy_signal_wait
        sell_signal = sell_signal_wait


    """
    define a dictionary with the confirmation function
    """
    confirmation_functions = {
        "buy": {
            "signal_0": signal_buy_cooldown,
            "signal_1": signal_buy_cooldown,         
            "signal_3": signal_no_wait,
            "signal_4": signal_no_wait,
            "signal_5": signal_buy_cooldown
        },
        "sell": {
            "signal_0": signal_sell_cci,
            "signal_1": signal_sell_cci,
            "signal_3": signal_no_wait,
            "signal_4": signal_no_wait,
            "signal_5": signal_sell_cci
        }
    }
    """
    If the position is waiting we need to check for confirmation
    """
    if position_manager.check_if_waiting():
        if trade_data["status"] == "buying":
            buy_signal = confirmation_functions["buy"]["signal_%i" %
                trade_data["signal_type"]](
                    position_manager, trade_data, indicators_data)
        elif trade_data["status"] == "selling":
            sell_signal = confirmation_functions["sell"]["signal_%i" %
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


    """
    Set a narrow stop loss if attempting to catch up the trend
    """

    if lift_sl and buy_signal and (bbands_below_keltner_low and
        current_price > bbands.select("bbands_middle")[-1]):
        stop_loss = price_to_percent(
            current_price, mid_low_point)

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

    with PlotScope.group("bayesian_bollinger", symbol):
        plot("sigma_up", sigma_probs_up)
        plot("sigma_down", sigma_probs_down)
        plot("prime_prob", prob_prime)

    with PlotScope.group("cooldown", symbol):
        plot("cooldown", int(cooldown))

    with PlotScope.group("hourly_mf1", symbol):
        if mfi_1h[-1]:
            plot("mfi_1h", mfi_1h[-1])

    with PlotScope.group("pnl", symbol):
        plot("pnl", float(state.positions_manager[
            symbol]["summary"]['pnl']))


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
        # if collect_data:
        #     position_manager.collect_data(state, indicators_data)
        if skip_buy is False:
            state.balance_quoted -= position_manager.position_value
            position_manager.open_market()
            position_manager.double_barrier(take_profit, stop_loss)
            if collect_data:
                position_manager.collect_data(state, indicators_data)
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

def get_signal_from_dict(signal_id, signal_dict):
    try:
        signal = signal_dict['%i' % signal_id]
    except KeyError:
        signal = False
    return signal

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
                    order_lower__price - current_price) / current_price
            except:
                success = False
        if success:
            self.cancel_stop_orders()
            self.double_barrier(
                take_profit, stop_loss, subtract_fees=subtract_fees)
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


def get_d(values, index, coef):
    try:
        d = sum(
        [coef["center"]["coefficients"][x] * values[
            index + coef["center"]["offsets"][x]] for x in range(
                len(coef["center"]["coefficients"]))])
    except IndexError:
        d = 1
    return d

def compute_derivs(x):
    coefficients_1_p4 = {
        "center": {
            "coefficients": [float(1/12), -float(2/3), 0, float(2/3), -float(1/12)],
            "offsets": [-2, -1, 0, 1, 2]

        },
        "forward": {
            "coefficients": [-float(25/12), 4, -3, float(4/3), -float(1/4)],
            "offsets": [0, 1, 2, 3, 4]
        },
        "backward": {
            "coefficients": [-float(1/3), float(3/2), -3, float(11/6)],
            "offsets": [-3, -2, -1, 0]
        }
    }
    coefficients_2_p4 = {
        "center": {
            "coefficients": [-float(1/12), float(4/3), -float(5/2), float(4/3), -float(1/12)],
            "offsets": [-2, -1, 0, 1, 2]
        },
        "forward": {
            "coefficients": [
                float(15/4), -float(77/6), float(107/6), -13, float(61/12), -float(5/6)],
            "offsets": [0, 1, 2, 3, 4, 5]
        },
        "backward": {
            "coefficients": [ -1., 4., -5., 2.],
            "offsets": [-3, -2, -1, 0]
        }
    }
    coefficients_1 = {
        "center": {
            "coefficients": [-0.5, 0. , 0.5],
            "offsets": [-1, 0, 1]
        },
        "forward": {
            "coefficients": [-1.5, 2. , -0.5],
            "offsets": [0, 1, 2]
        },
        "backward": {
            "coefficients": [ 0.5, -2. , 1.5],
            "offsets": [-2, -1, 0]
        }
    }
    coefficients_2 = {
        "center": {
            "coefficients": [1, -2 , 1],
            "offsets": [-1, 0, 1]
        },
        "forward": {
            "coefficients": [2., -5., 4., -1.],
            "offsets": [0, 1, 2, 3]
        },
        "backward": {
            "coefficients": [ -1., 4., -5., 2.],
            "offsets": [-3, -2, -1, 0]
        }
    }
    d1 = [get_d(x, i, coefficients_1) for i in range(len(x))]
    d2 = [get_d(x, i, coefficients_2) for i in range(len(x))]
    return(d1, d2)

def get_extrema(d1, d2, h, isMin):
  return [x for x in range(len(d1))
    if (d2[x] > 0 if isMin else d2[x] < 0) and
      (d1[x] == 0 or #slope is 0
        (x != len(d1) - 1 and #check next day
          (d1[x] > 0 and d1[x+1] < 0 and
           h[x] >= h[x+1] or
           d1[x] < 0 and d1[x+1] > 0 and
           h[x] <= h[x+1]) or
         x != 0 and #check prior day
          (d1[x-1] > 0 and d1[x] < 0 and
           h[x-1] < h[x] or
           d1[x-1] < 0 and d1[x] > 0 and
           h[x-1] > h[x])))]

def keltner_channels(data, period=20, atr_period=10, kc_mult=2, take_last=50):
    """
    calculate keltner channels mid, up and low values
    """
    ema = data.close.ema(period).select('ema')
    atr = data.atr(atr_period).select('atr')
    high = ema[-take_last:] + (kc_mult * atr[-take_last:])
    low = ema[-take_last:] - (kc_mult * atr[-take_last:])
    return {'middle': ema, 'high': high, 'low': low}


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

def compute_signal2(
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


def compute_signal(
    sigma_probs_up, sigma_probs_down, prob_prime,sigma_probs_up_prev,
    sigma_probs_down_prev, prob_prime_prev, lower_threshold=15, n_signals=4):
    buy_signal_record = {"0": False}
    sell_signal_record = {"0": False}
    small_offset = 0.001
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
        signal_1_sell = sigma_probs_down_prev == 0 and sigma_probs_down > (0 + small_offset)
        signal_1_buy = sigma_probs_up_prev == 0 and sigma_probs_up > (0 + small_offset)
        buy_signal_record["%i" % 1] = signal_1_buy
        sell_signal_record["%i" % 1] = signal_1_sell
        sell_using_sigma_probs_up.append(signal_1_sell)
        buy_using_sigma_probs_down.append(signal_1_buy)
    if 2 in n_signals:
        signal_2_sell = sigma_probs_down_prev < (1 - small_offset) and sigma_probs_down == 1
        signal_2_buy = sigma_probs_up_prev > (0 + small_offset) and sigma_probs_up == 0
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
        signal_5_buy = False
        if sigma_probs_up > small_offset and sigma_probs_up > small_offset and sigma_probs_up_prev > small_offset:
             signal_5_buy = sigma_probs_up > sigma_probs_down and sigma_probs_up > prob_prime and sigma_probs_up_prev > sigma_probs_up
        buy_signal_record["%i" % 5] = signal_5_buy
        sell_signal_record["%i" % 5] = signal_5_sell
        buy_using_sigma_probs_down.append(signal_5_buy)
        # sell_using_sigma_probs_up.append(
        #         sigma_probs_down_prev < 1 and sigma_probs_down == 1 and sigma_probs_down > sigma_probs_up and sigma_probs_down > prob_prime and sigma_probs_down)
    sell_signal = sell_using_prob_prime or any(sell_using_sigma_probs_up)
    buy_signal = buy_using_prob_prime or any(buy_using_sigma_probs_down)
    return (buy_signal, sell_signal, buy_signal_record, sell_signal_record)


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

def find_local_min(x):
    res = (np.diff(np.sign(np.diff(x))) == 2).nonzero()[0] + 1
    return res
def find_local_max(x):
    res = (np.diff(np.sign(np.diff(x))) == -2).nonzero()[0] + 1
    return res

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


def get_24h_candle(data):
    n_candles = 4 * 24 # based on 15m candles
    n_tot = len(data.times)
    op = None
    cl = None
    hi = None
    lo = None
    if n_tot < n_candles:
        print("too few candles")
        return
    for i in range(n_tot - 1, n_tot - n_candles, -1):
        if i == n_tot - 1:
            cl = float(data.close[i])
            hi = float(data.high[i])
            lo = float(data.low[i])
        else:
            op = float(data.open[i])
            if float(data.low[i]) < lo:
                lo = float(data.low[i])
            if float(data.high[i]) > hi:
                hi = float(data.high[i])
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
    resistance2 = pp + (yesterday_candle["high"] -yesterday_candle["low"])
    support2 = pp - (yesterday_candle["high"] -yesterday_candle["low"])
    resistance3 = pp + (2*(yesterday_candle["high"] -yesterday_candle["low"]))
    support3 = pp - (2*(yesterday_candle["high"] -yesterday_candle["low"]))
    return ({
        "resistance1": resistance1,
        "resistance2": resistance2,
        "resistance3": resistance3,
        "support1": support1,
        "support2": support2,
        "support3": support3
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
