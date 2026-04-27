#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Donchian channel provides clear breakout signals. 1d EMA filter ensures alignment with higher timeframe trend.
# Volume confirmation filters out false breakouts. Designed for ~20-30 trades/year per symbol.
# Works in both bull and bear markets by following the trend direction from 1d EMA.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian upper band + uptrend + volume
        if close[i] > highest_high[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
            if position != 1:  # Only signal on change
                signals[i] = 0.25
                position = 1
        
        # Short entry: price breaks below Donchian lower band + downtrend + volume
        elif close[i] < lowest_low[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
            if position != -1:  # Only signal on change
                signals[i] = -0.25
                position = -1
        
        # Exit: price crosses back through Donchian middle or trend reversal
        else:
            mid = (highest_high[i] + lowest_low[i]) / 2
            if position == 1 and (close[i] < mid or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (close[i] > mid or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0