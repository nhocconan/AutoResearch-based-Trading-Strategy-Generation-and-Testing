#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Donchian breakout provides clear entry/exit signals with defined risk.
# 12h EMA filter ensures we only trade in the direction of higher timeframe trend.
# Volume confirmation filters out false breakouts.
# Designed for 20-50 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian upper band in uptrend with volume
        if close[i] > highest_high[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
            if position != 1:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30  # maintain position
        # Short entry: price breaks below Donchian lower band in downtrend with volume
        elif close[i] < lowest_low[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
            if position != -1:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = -0.30  # maintain position
        # Exit: price crosses back through the middle of the Donchian channel
        elif position == 1 and close[i] < (highest_high[i] + lowest_low[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (highest_high[i] + lowest_low[i]) / 2:
            signals[i] = 0.0
            position = 0
        # Hold current position
        elif position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0