import pandas as pd
import numpy as np

def bands(close,window=50,no_of_stdev=1.5):

    rolling_mean = close.ewm(span=window).mean()
    rolling_std = close.ewm(span=window).std()
    upper_band = rolling_mean + (rolling_std * no_of_stdev)
    lower_band = rolling_mean - (rolling_std * no_of_stdev)

    long_signals = (close<=lower_band)
    short_signals = (close>=upper_band)

    side = pd.DataFrame(data=0, index=close.index, columns=['side'])

    side.loc[long_signals,'side'] = 1
    side.loc[short_signals,'side'] = -1

    return side,upper_band,lower_band

def cross_moving_average(close,l_window=50,s_window=20):

    mov_avg_l = close.rolling(window=l_window,min_periods=1).mean()
    mov_avg_s = close.rolling(window=s_window,min_periods=1).mean()

    mov_avg_l_shifft = close.shift(periods=1).rolling(window=l_window,min_periods=1).mean()
    mov_avg_s_shifft = close.shift(periods=1).rolling(window=s_window,min_periods=1).mean()

    long_signals = (mov_avg_l <= mov_avg_s) & (mov_avg_l_shifft > mov_avg_s_shifft)
    short_signals = (mov_avg_l >= mov_avg_s) & (mov_avg_l_shifft < mov_avg_s_shifft)

    side = pd.DataFrame(data=0, index=close.index, columns=['side'])

    side.loc[long_signals,'side'] = 1
    side.loc[short_signals,'side'] = -1

    return side,mov_avg_l,mov_avg_s


def rsi(close, periods = 14, ema = True):

    close_delta = close.diff()

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))

    return rsi