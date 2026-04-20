#!/usr/bin/env python3
"""
6h_1D_Pivot_R3S3_Fade_Signal
Hypothesis: Fade extreme price moves at 1d Camarilla R3/S3 levels with volume exhaustion and mean reversion.
In ranging markets (common in 2025 BTC/ETH), price reverts from extreme daily levels.
Short at R3 with bearish candle and low volume; long at S3 with bullish candle and low volume.
Exit at R2/S2 or opposite extreme. Works in both bull/bear as mean reversion persists.
Target: 50-100 total trades over 4 years (12-25/year) with position size 0.25.
"""

name = "6h_1D_Pivot_R3S3_Fade_Signal"
timeframe = "6h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    # Camarilla: Range = H - L
    # R4 = C + (H-L) * 1.1/2, R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4, S2 = C - (H-L) * 1.1/6
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first bar uses its own close
    
    r3_1d = close_prev + range_1d * 1.1 / 4
    s3_1d = close_prev - range_1d * 1.1 / 4
    r2_1d = close_prev + range_1d * 1.1 / 6
    s2_1d = close_prev - range_1d * 1.1 / 6
    
    # Align 1d levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume exhaustion: volume < 0.7 * 20-period average (low volume on test)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_exhaustion = volume < (0.7 * vol_ma20)
    
    # Candles: bullish = close > open, bearish = close < open
    open_prices = prices['open'].values
    bullish_candle = close > open_prices
    bearish_candle = close < open_prices
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Fade R3: short at resistance with bearish candle and low volume
            if close[i] > r3_1d_aligned[i] and bearish_candle[i] and volume_exhaustion[i]:
                signals[i] = -0.25
                position = -1
            # Fade S3: long at support with bullish candle and low volume
            elif close[i] < s3_1d_aligned[i] and bullish_candle[i] and volume_exhaustion[i]:
                signals[i] = 0.25
                position = 1
        
        elif position == -1:
            # Short exit: price reaches R2 (take profit) or breaks above S3 (stop)
            if close[i] <= r2_1d_aligned[i] or close[i] > s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        elif position == 1:
            # Long exit: price reaches S2 (take profit) or breaks below R3 (stop)
            if close[i] >= s2_1d_aligned[i] or close[i] < r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals