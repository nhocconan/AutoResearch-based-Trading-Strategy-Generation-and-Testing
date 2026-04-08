#!/usr/bin/env python3
"""
1D Donchian Breakout + Weekly Trend + Volume Confirmation
Hypothesis: Donchian(20) breakouts on daily timeframe capture strong trends.
Weekly EMA(21) filter ensures alignment with higher timeframe trend.
Volume confirmation (>1.5x 20-day average) filters weak breakouts.
Designed for low trade frequency (10-30/year) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
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
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_21_w = df_1w['close'].ewm(span=21, adjust=False).mean().values
    ema_21_w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_w)
    
    # Volume filter (>1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_21_w_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or weekly trend reverses
            if close[i] <= low_roll[i] or close[i] < ema_21_w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or weekly trend reverses
            if close[i] >= high_roll[i] or close[i] > ema_21_w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian high with trend alignment
            if (close[i] > high_roll[i] and 
                close[i] > ema_21_w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short below Donchian low with trend alignment
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_21_w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals