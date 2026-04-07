#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v2
Hypothesis: On 6h timeframe, use daily Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) combined with 12h EMA trend filter and volume confirmation. Fade extreme levels (R3/S3) in range markets, breakout at stronger levels (R4/S4) with trend alignment. Uses 12h EMA for trend and daily volume for confirmation. Designed for 12-37 trades/year (50-150 total over 4 years) to avoid fee drain while capturing mean reversion and breakout moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate daily Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_1d * 1.500
    r3 = close_1d + range_1d * 1.250
    s3 = close_1d - range_1d * 1.250
    s4 = close_1d - range_1d * 1.500
    
    # Align levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA direction
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or strong breakout above R4 fails
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or strong breakdown below S4 fails
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Fade at extreme levels (R3/S3) - mean reversion
                if close[i] >= r3_aligned[i] and close[i] < r4_aligned[i]:
                    # Sell at R3, expecting reversion to pivot
                    position = -1
                    signals[i] = -0.25
                elif close[i] <= s3_aligned[i] and close[i] > s4_aligned[i]:
                    # Buy at S3, expecting reversion to pivot
                    position = 1
                    signals[i] = 0.25
                # Breakout at stronger levels (R4/S4) with trend alignment
                elif close[i] > r4_aligned[i] and uptrend:
                    # Breakout above R4 in uptrend
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i] and downtrend:
                    # Breakdown below S4 in downtrend
                    position = -1
                    signals[i] = -0.25
    
    return signals