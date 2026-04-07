#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_ema_volume_v3
Hypothesis: Camarilla pivot levels from 1d: fade at R3/S3 with volume confirmation and EMA20 trend filter.
In ranging markets, price reverts from R3/S3. In trending markets, breaks of R4/S4 with volume and trend alignment indicate continuation.
Uses strict entry conditions to limit trades (target: 12-37/year) and avoid fee crush.
Works in bull/bear by adapting to market structure via volume and trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_ema_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average (stricter)
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema20 = close[i] > ema20_1d_aligned[i]
        below_ema20 = close[i] < ema20_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (mean reversion) or trend turns bearish with volume
            if close[i] <= s3_aligned[i] or (below_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 (mean reversion) or trend turns bullish with volume
            if close[i] >= r3_aligned[i] or (above_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3: sell at R3, buy at S3 in ranging markets
            # But only if volume confirms and trend is not strong
            if close[i] >= r3_aligned[i] and vol_spike and not above_ema20:
                # Potential short at R3 rejection
                position = -1
                signals[i] = -0.25
            elif close[i] <= s3_aligned[i] and vol_spike and not below_ema20:
                # Potential long at S3 bounce
                position = 1
                signals[i] = 0.25
            # Breakout continuation at R4/S4 with volume and trend
            elif close[i] > r4_aligned[i] and vol_spike and above_ema20:
                # Bullish breakout with volume and trend
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_aligned[i] and vol_spike and below_ema20:
                # Bearish breakout with volume and trend
                position = -1
                signals[i] = -0.25
    
    return signals