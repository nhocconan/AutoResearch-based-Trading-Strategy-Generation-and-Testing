#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Donchian(20) provides clear breakout levels with strong trend-following properties.
# 1d EMA34 filters trades to only take in the direction of the daily trend.
# Volume confirmation ensures breakouts are supported by participation.
# Exits on opposite Donchian(10) breakout to capture trends while limiting drawdown.
# Targets 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian channels (10-period for exit)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian(20) high, 1d uptrend, volume confirmation
            if close[i] > high_20[i] and close[i] > ema34_1d_aligned[i] and vol_conf[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian(20) low, 1d downtrend, volume confirmation
            elif close[i] < low_20[i] and close[i] < ema34_1d_aligned[i] and vol_conf[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian(10) low
            if close[i] < low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian(10) high
            if close[i] > high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals