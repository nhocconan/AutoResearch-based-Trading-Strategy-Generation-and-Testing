#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout + Volume Confirmation v1
Hypothesis: Weekly trend filter (using 1-week Donchian channels) combined with daily price action and volume confirmation reduces whipsaws while capturing major trends. The weekly trend ensures we only trade in the direction of the higher timeframe momentum, while daily Donchian breakouts provide precise entry/exit points. Volume confirmation filters out false breakouts. This approach works in both bull and bear markets by adapting to the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-period) for trend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly upper/lower bands (20-period lookback)
    weekly_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Daily Donchian breakout channels (20-period)
    daily_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    daily_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(daily_high[i]) or np.isnan(daily_low[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below daily lower band or weekly trend turns bearish
            if close[i] <= daily_low[i] or close[i] < weekly_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above daily upper band or weekly trend turns bullish
            if close[i] >= daily_high[i] or close[i] > weekly_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with weekly uptrend and volume
            if (close[i] >= daily_high[i] and 
                close[i] > weekly_high_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with weekly downtrend and volume
            elif (close[i] <= daily_low[i] and 
                  close[i] < weekly_low_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals