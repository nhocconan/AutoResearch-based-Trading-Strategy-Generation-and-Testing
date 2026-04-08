#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout + Volume Confirmation v1
Hypothesis: Weekly Donchian breakouts with volume confirmation and 1d trend filter (EMA 200) capture major trends while avoiding whipsaws. The 1d timeframe targets 10-30 trades/year, minimizing fee drag. Volume ensures breakout validity, and EMA 200 avoids counter-trend trades. Designed to work in both bull and bear markets by filtering with long-term trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-period high/low)
    high_20 = pd.Series(df_weekly['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_weekly['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_weekly, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_weekly, low_20)
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_200_1d[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly low or trend reverses
            if close[i] <= low_20_aligned[i] or close[i] < ema_200_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly high or trend reverses
            if close[i] >= high_20_aligned[i] or close[i] > ema_200_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] >= high_20_aligned[i] and 
                close[i] > ema_200_1d[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] <= low_20_aligned[i] and 
                  close[i] < ema_200_1d[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals