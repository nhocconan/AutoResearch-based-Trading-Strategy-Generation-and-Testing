#!/usr/bin/env python3
"""
6H Weekly Donchian Breakout + Daily Trend Filter
Hypothesis: Weekly Donchian(20) breakouts capture major trend moves. Daily EMA(50) filters for trend alignment.
Volume confirmation ensures breakouts are genuine. Designed for 6h timeframe to capture fewer, higher-quality trades.
Works in bull markets (breakouts up) and bear markets (breakdowns down). Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_breakout_daily_trend"
timeframe = "6h"
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
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian(20) - using high/low of past 20 weekly bars
    # Using rolling window on weekly high/low
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe (weekly levels known at open)
    donchian_high_6h = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Daily EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter (>1.3x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low or trend reverses
            if close[i] <= donchian_low_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high or trend reverses
            if close[i] >= donchian_high_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above weekly Donchian high with trend alignment
            if (close[i] > donchian_high_6h[i] and 
                close[i] > ema_50_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakdown short below weekly Donchian low with trend alignment
            elif (close[i] < donchian_low_6h[i] and 
                  close[i] < ema_50_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals