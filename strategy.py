#!/usr/bin/env python3
"""
1D Donchian Breakout + Weekly Trend + Volume Confirmation v1
Hypothesis: Donchian(20) breakouts from daily timeframe capture strong momentum.
Breakouts above 20-day high or below 20-day low with weekly EMA trend alignment and volume confirmation.
Designed for 1d timeframe to balance trade frequency and signal quality in both bull and bear markets.
Target: 7-25 trades/year per signal.
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
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channels (20-day high/low) from previous day
    donchian_high = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1)
    donchian_low = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1)
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_21 = df_1w['close'].ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Align to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high.values)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low.values)
    ema_21_1d = align_htf_to_ltf(prices, df_1w, ema_21.values)
    
    # Volume filter (>1.5x 20-period average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema_21_1d[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= donchian_low_1d[i] or close[i] < ema_21_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= donchian_high_1d[i] or close[i] > ema_21_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at Donchian high with trend alignment
            if (close[i] >= donchian_high_1d[i] and 
                close[i] > ema_21_1d[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at Donchian low with trend alignment
            elif (close[i] <= donchian_low_1d[i] and 
                  close[i] < ema_21_1d[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals