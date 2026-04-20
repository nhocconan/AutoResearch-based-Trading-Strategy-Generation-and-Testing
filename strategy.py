#!/usr/bin/env python3
"""
4h_Donchian20_VolumeBreakout_Camilla24h_Filter
Hypothesis: Trade 4h Donchian(20) breakouts with volume confirmation and 24h CAMARILLA R3/S3 filter.
Long when price breaks above Donchian upper band with volume > 1.8x 20-period average and close > 24h CAMARILLA R3.
Short when breaks below Donchian lower band with volume spike and close < 24h CAMARILLA S3.
Exit when price reverts to Donchian middle band (mean of upper/lower) or volume dries up.
Designed to capture strong momentum bursts in both bull and bear markets while avoiding false breakouts in chop.
Target: 80-160 total trades over 4 years (20-40/year) with position size 0.25.
Uses volume spike to filter weak breakouts and CAMARILLA levels to ensure breakout occurs beyond key intraday resistance/support.
Works in bull/bear: volume filter adapts to volatility regime, CAMARILLA levels provide dynamic support/resistance.
"""

name = "4h_Donchian20_VolumeBreakout_Camilla24h_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 24h data ONCE before loop for CAMARILLA levels
    df_24h = get_htf_data(prices, '24h')
    if len(df_24h) < 2:
        return np.zeros(n)
    
    # Calculate 24h CAMARILLA levels (using prior 24h bar's high, low, close)
    high_24h = df_24h['high'].values
    low_24h = df_24h['low'].values
    close_24h = df_24h['close'].values
    
    # CAMARILLA formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We only need R3 and S3
    range_24h = high_24h - low_24h
    r3_24h = close_24h + range_24h * 1.1 / 4
    s3_24h = close_24h - range_24h * 1.1 / 4
    
    # Align 24h CAMARILLA levels to 4h timeframe
    r3_24h_aligned = align_htf_to_ltf(prices, df_24h, r3_24h)
    s3_24h_aligned = align_htf_to_ltf(prices, df_24h, s3_24h)
    
    # Calculate Donchian(20) channels
    lookback = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    middle = np.full_like(close, np.nan)
    
    for i in range(lookback, n):
        upper[i] = np.max(high[i-lookback:i])
        lower[i] = np.min(low[i-lookback:i])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Calculate volume filter (volume > 1.8x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready (20 for Donchian + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(r3_24h_aligned[i]) or np.isnan(s3_24h_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume filter AND close > 24h R3
            if close[i] > upper[i] and volume_filter[i] and close[i] > r3_24h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band with volume filter AND close < 24h S3
            elif close[i] < lower[i] and volume_filter[i] and close[i] < s3_24h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian middle band OR volume dries up
            if close[i] < middle[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian middle band OR volume dries up
            if close[i] > middle[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals