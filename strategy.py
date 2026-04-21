#!/usr/bin/env python3
"""
6h_PivotPoint_R3S3_Fade_V1
Hypothesis: Fade extreme Camarilla pivot levels (R3/S3) on 6b with 1d trend filter (EMA50) and volume confirmation. In ranging markets, price reverts from R3/S3; in trending markets, 1d EMA50 filter prevents counter-trend fades. Works in both bull (fades at R3 in uptrend blocked) and bear (fades at S3 in downtrend blocked) by requiring 1d trend alignment. Target 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for EMA50 trend filter
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for Trend Filter ===
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # We need the previous completed 1d bar's OHLC
    # Since we're on 6h timeframe, we can approximate using rolling window
    # But better: use get_htf_data for 1d and calculate pivots there
    
    # Recalculate: get 1d OHLC for pivot calculation
    # We already have df_1d from get_htf_data
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    # We'll use H3/L3 as R3/S3 for fade
    
    # But we need to align these to 6h bars
    # So calculate on 1d then align
    
    # Actually, let's simplify: use the 1d bar's high/low to calculate R3/S3
    # and align to 6h timeframe
    
    # We'll calculate the pivot levels from the 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    r3_1d = close_1d + 1.1 * (high_1d - low_1d)
    s3_1d = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long fade at S3: price < S3 and 1d uptrend (price > EMA50)
            if price < s3_aligned[i] and ema_1d_aligned[i] < price and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short fade at R3: price > R3 and 1d downtrend (price < EMA50)
            elif price > r3_aligned[i] and ema_1d_aligned[i] > price and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses EMA50 or reaches opposite level (R3)
            if price > ema_1d_aligned[i] or price > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses EMA50 or reaches opposite level (S3)
            if price < ema_1d_aligned[i] or price < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PivotPoint_R3S3_Fade_V1"
timeframe = "6h"
leverage = 1.0