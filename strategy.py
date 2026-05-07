#!/usr/bin/env python3
# 12h_Donchian20_1dTrend_VolumeBreakout
# Hypothesis: Uses 12h Donchian breakout with 1d trend filter (EMA50) and volume confirmation.
# Enters long when price breaks above 12h Donchian high(20) with 1d uptrend (price > EMA50) and volume spike.
# Enters short when price breaks below 12h Donchian low(20) with 1d downtrend (price < EMA50) and volume spike.
# 12h timeframe reduces trade frequency to avoid fee drag; Donchian provides objective breakout levels;
# 1d trend filter ensures alignment with intermediate trend; volume confirms breakout strength.
# Targets 12-30 trades/year on 12h timeframe.

name = "12h_Donchian20_1dTrend_VolumeBreakout"
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
    
    # Get 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter on 12h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high, above 1d EMA50 trend, volume spike
            if close[i] > high_20[i] and close[i] > ema_50_1d_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian low, below 1d EMA50 trend, volume spike
            elif close[i] < low_20[i] and close[i] < ema_50_1d_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below 12h Donchian low or below 1d EMA50
            if close[i] < low_20[i] or close[i] < ema_50_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above 12h Donchian high or above 1d EMA50
            if close[i] > high_20[i] or close[i] > ema_50_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals