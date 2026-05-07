#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: 4-hour Donchian channel breakouts filtered by 1-day trend direction
# (above/below EMA34) and volume confirmation reduce false signals while capturing
# strong momentum moves. Works in both bull and bear markets by only trading in
# direction of higher timeframe trend. Target: 20-30 trades/year.

name = "4h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4-hour Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-hour volume average (20-period) for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high with uptrend and volume
            if close[i] > high_max_20[i] and close[i] > ema_34_4h[i] and volume[i] > (1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with downtrend and volume
            elif close[i] < low_min_20[i] and close[i] < ema_34_4h[i] and volume[i] > (1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below 20-period low or trend reversal
            if close[i] < low_min_20[i] or close[i] < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above 20-period high or trend reversal
            if close[i] > high_max_20[i] or close[i] > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals