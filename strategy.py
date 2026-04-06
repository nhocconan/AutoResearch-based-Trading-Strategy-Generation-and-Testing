#!/usr/bin/env python3
"""
1h Donchian Breakout with 4h Trend Filter and Volume Confirmation
Hypothesis: Uses 1h for precise entry/exit timing while relying on 4h trend direction (via Donchian breakout) and volume confirmation to filter false signals. Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe. Works in bull markets via long breakouts and bear markets via short breakdowns. Includes session filter (08-20 UTC) to reduce noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_4htrend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    highest_high_4h = np.full_like(high_4h, np.nan)
    lowest_low_4h = np.full_like(low_4h, np.nan)
    for i in range(20, len(high_4h)):
        highest_high_4h[i] = np.max(high_4h[i-20:i])
        lowest_low_4h[i] = np.min(low_4h[i-20:i])
    
    # 4h trend: 1 if price > upper band, -1 if price < lower band, 0 otherwise
    trend_4h = np.zeros_like(close_4h)
    for i in range(20, len(close_4h)):
        if close_4h[i] > highest_high_4h[i]:
            trend_4h[i] = 1
        elif close_4h[i] < lowest_low_4h[i]:
            trend_4h[i] = -1
    
    # Align 4h trend to 1h (shifted by 1 bar inside align_htf_to_ltf)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter (20-period average)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08-20 UTC (pre-compute)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup
    start = 20
    
    for i in range(start, n):
        # Skip if 4h trend not ready
        if np.isnan(trend_4h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        vol_condition = not np.isnan(vol_ma[i]) and volume[i] > vol_ma[i] * 1.5
        
        # Session condition
        sess_condition = session_filter[i]
        
        # Check exits
        if position == 1:  # long
            # Exit: 1h close below 20-period low OR trend flips to bear
            if i >= 20:
                lowest_low_1h = np.min(low[i-20:i])
                if close[i] < lowest_low_1h or trend_4h_aligned[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20
        elif position == -1:  # short
            # Exit: 1h close above 20-period high OR trend flips to bull
            if i >= 20:
                highest_high_1h = np.max(high[i-20:i])
                if close[i] > highest_high_1h or trend_4h_aligned[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 1h breakout in direction of 4h trend + volume + session
            if i >= 20:
                highest_high_1h = np.max(high[i-20:i])
                lowest_low_1h = np.min(low[i-20:i)]
                
                bull_breakout = close[i] > highest_high_1h
                bear_breakout = close[i] < lowest_low_1h
                
                if bull_breakout and trend_4h_aligned[i] == 1 and vol_condition and sess_condition:
                    signals[i] = 0.20
                    position = 1
                elif bear_breakout and trend_4h_aligned[i] == -1 and vol_condition and sess_condition:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals