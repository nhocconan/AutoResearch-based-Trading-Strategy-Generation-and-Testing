#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R3S3_Fade_With_Volume_Filter
Hypothesis: Fade at Camarilla R3/S3 levels (strong intraday support/resistance) on 6h chart with volume confirmation.
In ranging markets, price reverts from R3/S3 toward the mean. In trending markets, breaks of R3/S3 with volume signal continuation.
Uses 12h EMA for trend filter to distinguish between mean-reversion and breakout contexts.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar: 
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # R4 = close + 1.5*(high-low)/2, S4 = close - 1.5*(high-low)/2
    hl_range = high_12h - low_12h
    r3 = close_12h + 1.1 * hl_range / 2
    s3 = close_12h - 1.1 * hl_range / 2
    r4 = close_12h + 1.5 * hl_range / 2
    s4 = close_12h - 1.5 * hl_range / 2
    
    # Align Camarilla levels to 6h timeframe (no extra delay needed for pivot points)
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    r4_6h = align_htf_to_ltf(prices, df_12h, r4)
    s4_6h = align_htf_to_ltf(prices, df_12h, s4)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: >1.6x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.6 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Fade at R3/S3 in ranging/weak trend: short at R3, long at S3
            if price >= r3_6h[i] and price < r4_6h[i] and ema34 > price * 0.98:  # Below or slightly above EMA
                signals[i] = -0.25
                position = -1
            elif price <= s3_6h[i] and price > s4_6h[i] and ema34 < price * 1.02:  # Above or slightly below EMA
                signals[i] = 0.25
                position = 1
            # Breakout with volume: continue trend if price breaks R4/S4 with volume spike
            elif price > r4_6h[i] and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            elif price < s4_6h[i] and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: mean reversion to S3 or trend breakdown
            if price <= s3_6h[i] or (price < ema34 and price < r3_6h[i]):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: mean reversion to R3 or trend reversal
            if price >= r3_6h[i] or (price > ema34 and price > s3_6h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_Pivot_R3S3_Fade_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0