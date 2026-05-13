#!/usr/bin/env python3
# Hypothesis: 1d Donchian channel breakout with weekly trend filter and volume confirmation.
# In bull markets, price breaks above upper Donchian(20); in bear markets, breaks below lower Donchian(20).
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation avoids false breakouts. Designed for low trade frequency (<25/year) to minimize fee drag.
# Works in both bull and bear markets by capturing breakouts in direction of weekly trend.

name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Calculate 1d Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    # Get weekly trend using SMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    sma50_1w = close_1w_series.rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with weekly uptrend and volume
            if close[i] > upper_channel[i] and close[i] > sma50_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with weekly downtrend and volume
            elif close[i] < lower_channel[i] and close[i] < sma50_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back below upper Donchian
            if close[i] < upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back above lower Donchian
            if close[i] > lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals