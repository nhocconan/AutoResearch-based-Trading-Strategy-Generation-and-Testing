#!/usr/bin/env python3
"""
1D Donchian Breakout + Weekly Trend + Volume Confirmation
Hypothesis: Daily Donchian(20) breakouts aligned with weekly trend (EMA21) and volume capture strong momentum.
Breakouts above upper band or below lower band with weekly EMA trend alignment and volume >1.5x average trigger entries.
Designed for 1d timeframe to capture major moves with low trade frequency (<25/year) suitable for both bull and bear markets.
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(21) for trend filter
    ema_21 = df_1w['close'].ewm(span=21, adjust=False).mean().values
    ema_21_1d = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_21_1d[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian band
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian band
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above upper band with trend alignment
            if (close[i] >= high_20[i] and 
                close[i] > ema_21_1d[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short below lower band with trend alignment
            elif (close[i] <= low_20[i] and 
                  close[i] < ema_21_1d[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals