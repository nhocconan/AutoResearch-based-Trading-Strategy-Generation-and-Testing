#!/usr/bin/env python3
"""
6h_HTF_1d_Camarilla_R3S3_Fade_v1
Hypothesis: On 6h timeframe, fade extreme Camarilla pivot levels (R3/S3) from prior 1d when price shows rejection (close near open) with volume confirmation. 
In ranging markets (2025+), price often reverses at these levels. Uses 1d HTF for pivot calculation, aligned to 6h bars.
Works in both bull and bear by fading extremes rather than following trends.
Target: 12-37 trades/year (50-150 over 4 years). Position size: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots using prior 1d bar (HLC of previous day)
    # Camarilla: R3 = Close + (High - Low) * 1.1/2, S3 = Close - (High - Low) * 1.1/2
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_range = prev_high - prev_low
    r3 = prev_close + pivot_range * 1.1 / 2.0
    s3 = prev_close - pivot_range * 1.1 / 2.0
    
    # Align to 6h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators ===
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        
        # Price rejection conditions: close near open (small body)
        body_size = abs(close[i] - open_price[i])
        candle_range = high[i] - low[i]
        if candle_range > 0:
            body_ratio = body_size / candle_range  # 0 = doji, 1 = strong body
            rejection = body_ratio < 0.3  # small body relative to range
        else:
            rejection = False
        
        if position == 0:
            # Long fade at S3: price touches/below S3 and shows rejection
            if price <= s3_aligned[i] and rejection and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short fade at R3: price touches/above R3 and shows rejection
            elif price >= r3_aligned[i] and rejection and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price moves back above midpoint or strong reversal
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves back below midpoint or strong reversal
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_Camarilla_R3S3_Fade_v1"
timeframe = "6h"
leverage = 1.0