#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend_Filter
Hypothesis: Trade 12h Donchian breakouts with volume confirmation and daily trend filter.
Long when price breaks above 20-period upper band + volume surge + daily uptrend.
Short when price breaks below 20-period lower band + volume surge + daily downtrend.
Exit when price returns to midpoint of the Donchian channel.
This captures strong trends while filtering weak breakouts with volume and trend alignment.
Works in bull/bear: daily trend filter prevents counter-trend trades, volume confirms breakout strength.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "12h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_daily = df_daily['close'].values
    ema20_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_daily[19] = np.mean(close_daily[:20])
        for i in range(20, len(close_daily)):
            ema20_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema20_daily[i-1]
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    
    # Calculate 12-period Donchian channels
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    
    for i in range(19, n):
        upper_band[i] = np.max(high[i-19:i+1])
        lower_band[i] = np.min(low[i-19:i+1])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    # Calculate 20-period volume average for volume spike detection
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema20_daily_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        volume_surge = volume[i] > 1.5 * vol_ma[i]  # Volume 50% above average
        
        if position == 0:
            # Long: break above upper band + volume surge + daily uptrend
            if close[i] > upper_band[i] and volume_surge and close[i] > ema20_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + volume surge + daily downtrend
            elif close[i] < lower_band[i] and volume_surge and close[i] < ema20_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band
            if close[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band
            if close[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals