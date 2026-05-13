# 12h_1dWMA_Cross_Strategy
# Hypothesis: 12-hour WMA crossovers with 1-day trend filter capture medium-term momentum.
# Long when 12h WMA(9) crosses above WMA(21) with 1d WMA(55) uptrend.
# Short when 12h WMA(9) crosses below WMA(21) with 1d WMA(55) downtrend.
# Uses volume confirmation (>1.5x 20-period average) to filter false signals.
# Target: 15-25 trades/year per symbol to minimize fee drag.
# Works in bull markets via trend following and in bear markets via short signals during downtrends.

name = "12h_1dWMA_Cross_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h WMA(9) and WMA(21) for crossover signals
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMA using pandas for efficiency with NaN handling
    close_series = pd.Series(close)
    wma9 = close_series.rolling(window=9, min_periods=9).apply(
        lambda x: np.dot(x, np.arange(1, 10)) / 45, raw=True
    ).values
    wma21 = close_series.rolling(window=21, min_periods=21).apply(
        lambda x: np.dot(x, np.arange(1, 22)) / 231, raw=True
    ).values
    
    # 1d WMA(55) for trend filter (using HTF data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    wma55_1d = pd.Series(close_1d).rolling(window=55, min_periods=55).apply(
        lambda x: np.dot(x, np.arange(1, 56)) / 1540, raw=True
    ).values
    wma55_1d_aligned = align_htf_to_ltf(prices, df_1d, wma55_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):  # Start after warmup for WMA21 and WMA55
        if position == 0:
            # LONG: WMA9 crosses above WMA21, 1d WMA55 uptrend, volume confirmation
            if (wma9[i] > wma21[i] and wma9[i-1] <= wma21[i-1] and 
                close[i] > wma55_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: WMA9 crosses below WMA21, 1d WMA55 downtrend, volume confirmation
            elif (wma9[i] < wma21[i] and wma9[i-1] >= wma21[i-1] and 
                  close[i] < wma55_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WMA9 crosses below WMA21 or trend turns down
            if wma9[i] < wma21[i] and wma9[i-1] >= wma21[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WMA9 crosses above WMA21 or trend turns up
            if wma9[i] > wma21[i] and wma9[i-1] <= wma21[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals