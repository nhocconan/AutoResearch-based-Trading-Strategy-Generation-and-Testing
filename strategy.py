#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout + Volume Confirmation v1
Hypothesis: Weekly Donchian breakouts with volume confirmation and trend filter work in both bull and bear markets by capturing major trend continuations while avoiding whipsaws. The 1d timeframe targets 10-25 trades/year, minimizing fee drag. Volume ensures breakout validity, and weekly trend filter prevents counter-trend entries.
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
    
    # Weekly data for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend filter
    ema_50_weekly = df_weekly['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Weekly Donchian channels (20-period high/low)
    high_series = pd.Series(df_weekly['high'].values)
    low_series = pd.Series(df_weekly['low'].values)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_weekly_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low or trend reverses
            if close[i] <= donchian_low_aligned[i] or close[i] < ema_50_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high or trend reverses
            if close[i] >= donchian_high_aligned[i] or close[i] > ema_50_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] >= donchian_high_aligned[i] and 
                close[i] > ema_50_weekly_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] <= donchian_low_aligned[i] and 
                  close[i] < ema_50_weekly_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals