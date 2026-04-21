#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_TrendFilter_V1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter (EMA34) and volume confirmation.
Long when price breaks above 20-day high + weekly EMA34 up + volume spike.
Short when price breaks below 20-day low + weekly EMA34 down + volume spike.
Exit when price crosses 20-day midpoint.
Works in bull by following weekly trend, avoids bear traps via trend filter.
Target: 10-25 trades/year per symbol (low frequency reduces fee drag).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's 20-period Donchian channels (using prior 20 days)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    mid_20 = (high_20 + low_20) / 2
    
    # Align to daily timeframe (1d is same as price timeframe)
    high_20_aligned = high_20  # No alignment needed for same timeframe
    low_20_aligned = low_20
    mid_20_aligned = mid_20
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA34 on weekly
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter: EMA34 slope
        if i >= 51:
            ema34_prev = ema34_1w_aligned[i-1]
            ema34_curr = ema34_1w_aligned[i]
            weekly_uptrend = ema34_curr > ema34_prev
            weekly_downtrend = ema34_curr < ema34_prev
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long conditions: break above 20-day high + volume + weekly uptrend
            if price > high_20_aligned[i] and volume_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below 20-day low + volume + weekly downtrend
            elif price < low_20_aligned[i] and volume_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below 20-day midpoint
            if price < mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above 20-day midpoint
            if price > mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian_Breakout_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0