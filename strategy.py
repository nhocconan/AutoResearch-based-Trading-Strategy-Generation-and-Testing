#!/usr/bin/env python3
# 4h_1d_Donchian_20_Breakout_Volume_Trend
# Hypothesis: Trade 4h Donchian(20) breakouts with volume confirmation and 1d trend filter.
# In bull markets: buy breakouts above upper band with rising volume and 1d uptrend.
# In bear markets: sell breakdowns below lower band with rising volume and 1d downtrend.
# Uses 1d EMA50 as trend filter to avoid counter-trend trades. Targets 20-50 trades/year.
# Volume surge (>1.5x 20-bar average) confirms breakout strength.

name = "4h_1d_Donchian_20_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above upper Donchian with volume surge and 1d uptrend
            if (close[i] > high_max[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian with volume surge and 1d downtrend
            elif (close[i] < low_min[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below lower Donchian
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above upper Donchian
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals