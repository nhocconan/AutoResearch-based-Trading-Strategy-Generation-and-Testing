#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_Volume_Trend
Hypothesis: Use weekly Donchian channel breakouts with volume confirmation and daily trend filter.
Long when price breaks above weekly Donchian upper with volume > 1.5x 20-day avg AND price > daily EMA20.
Short when price breaks below weekly Donchian lower with volume > 1.5x 20-day avg AND price < daily EMA20.
Exit when price crosses back through weekly Donchian midpoint.
Designed for 1d timeframe to capture weekly trends with ~10-20 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume and trend filters reduce false breakouts and whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly Donchian channel (20-period)
    lookback = 20
    upper = np.full_like(high_weekly, np.nan)
    lower = np.full_like(low_weekly, np.nan)
    
    for i in range(lookback, len(high_weekly)):
        upper[i] = np.max(high_weekly[i-lookback:i])
        lower[i] = np.min(low_weekly[i-lookback:i])
    
    # Midpoint for exit
    midpoint = (upper + lower) / 2.0
    
    # Align to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower)
    midpoint_aligned = align_htf_to_ltf(prices, df_weekly, midpoint)
    
    # Daily EMA20 for trend filter
    close_s = prices['close']
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(ema_20[i])):
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
        
        if position == 0:
            # Long conditions: break above weekly upper + volume confirmation + price above EMA20
            if price > upper_aligned[i] and volume_ok and price > ema_20[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below weekly lower + volume confirmation + price below EMA20
            elif price < lower_aligned[i] and volume_ok and price < ema_20[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly midpoint
            if price < midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly midpoint
            if price > midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0