#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets. 1d EMA34 filters counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above upper Donchian band in uptrend with volume
        if close[i] > high_20[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short entry: price breaks below lower Donchian band in downtrend with volume
        elif close[i] < low_20[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle of channel
        elif position == 1 and close[i] < (high_20[i] + low_20[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (high_20[i] + low_20[i]) / 2:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0