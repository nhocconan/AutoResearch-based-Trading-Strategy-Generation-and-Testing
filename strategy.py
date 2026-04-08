#!/usr/bin/env python3
"""
1d Donchian Channel Breakout + Weekly Trend + Volume Confirmation v1
Hypothesis: Daily Donchian breakouts filtered by weekly EMA trend and volume capture sustained moves while avoiding whipsaws. The daily timeframe targets 5-15 trades/year, minimizing fee drag. Volume validates breakout strength, and weekly trend ensures alignment with higher-timeframe momentum, working in both bull and bear regimes by adapting to volatility regimes.
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
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(20) for trend filter
    ema_20_weekly = df_weekly['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily Donchian Channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_weekly_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend reverses
            if close[i] <= low_min[i] or close[i] < ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend reverses
            if close[i] >= high_max[i] or close[i] > ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] >= high_max[i] and 
                close[i] > ema_20_weekly_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] <= low_min[i] and 
                  close[i] < ema_20_weekly_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals