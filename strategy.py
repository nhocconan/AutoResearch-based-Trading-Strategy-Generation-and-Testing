#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume and Trend Filter
Hypothesis: Weekly Donchian breakouts capture major trends. Filter by daily EMA trend and volume confirmation. Works in bull/bear by using volatility-adjusted position sizing and trend alignment. Targets 10-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian Channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    highest_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # Daily EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low OR trend reverses
            if (close[i] <= lowest_low_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high OR trend reverses
            if (close[i] >= highest_high_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] > highest_high_aligned[i-1] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] < lowest_low_aligned[i-1] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals