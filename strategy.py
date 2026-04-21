#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R3S3_Fade_Reverse_V1
Hypothesis: On 6h timeframe, use 1d Camarilla pivot levels (R3/S3) for mean reversion entries in ranging markets. Enter short at R3 with volume confirmation, long at S3 with volume confirmation. Exit when price reaches opposite pivot level (S3 for shorts, R3 for longs) or on reversal signal. Uses 6h EMA34 as trend filter to avoid trading against strong trends. Designed for low-frequency, high-conviction mean reversion in both bull and bear markets where price respects daily pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate ranges
    rng = high_1d - low_1d
    # Camarilla levels
    r3 = pp + rng * 1.1 / 4.0
    s3 = pp - rng * 1.1 / 4.0
    r4 = pp + rng * 1.1 / 2.0
    s4 = pp - rng * 1.1 / 2.0
    
    # Align to 6h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA34 for trend filter (avoid strong trends)
    close_s = pd.Series(close)
    ema34 = close_s.ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Volume confirmation (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        
        # Trend filter: only mean revert when not in strong trend
        # In strong uptrend, avoid shorts; in strong downtrend, avoid longs
        strong_uptrend = price > ema34[i] * 1.02  # 2% above EMA34
        strong_downtrend = price < ema34[i] * 0.98  # 2% below EMA34
        
        if position == 0:
            # Long entry: price at S3 with volume confirmation, not in strong downtrend
            if price <= s3_aligned[i] and vol_ok and not strong_downtrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price at R3 with volume confirmation, not in strong uptrend
            elif price >= r3_aligned[i] and vol_ok and not strong_uptrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: price reaches R3 (opposite level) or reversal signal
            if price >= r3_aligned[i] or (price < ema34[i] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price reaches S3 (opposite level) or reversal signal
            if price <= s3_aligned[i] or (price > ema34[i] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_R3S3_Fade_Reverse_V1"
timeframe = "6h"
leverage = 1.0