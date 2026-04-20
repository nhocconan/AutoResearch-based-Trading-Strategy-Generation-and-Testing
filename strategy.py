#!/usr/bin/env python3
# 12h_1W_Donchian_Breakout_Volume_TrendFilter_v1
# Hypothesis: On 12h timeframe, trade breakouts of weekly Donchian channels (20-period) with volume confirmation and weekly EMA trend filter.
# In bull markets, buy breakouts above upper band; in bear markets, sell breakdowns below lower band.
# Weekly EMA20 filters trend direction: only long when price > EMA20, short when price < EMA20.
# Volume must be above 1.5x 20-period average to confirm breakout strength.
# Targets 15-30 trades/year by requiring confluence of breakout, volume, and trend filter.

name = "12h_1W_Donchian_Breakout_Volume_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper and lower bands
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above upper Donchian with volume and trend filter
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_20_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below lower Donchian with volume and trend filter
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_20_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA20 or breakdown below lower Donchian
            if close[i] < ema_20_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 or breakout above upper Donchian
            if close[i] > ema_20_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals