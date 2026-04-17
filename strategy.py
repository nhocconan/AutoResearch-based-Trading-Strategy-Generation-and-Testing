#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot R1/S1 breakout + volume confirmation + 12h EMA trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.3x average AND 12h EMA34 > EMA50.
Short when price breaks below Camarilla S1 AND volume > 1.3x average AND 12h EMA34 < EMA50.
Exit when price reverts to Camarilla midpoint (close) OR 12h EMA flips.
Uses 4h for pivot calculation, 12h for trend filter to reduce whipsaw and capture medium-term momentum.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide high-probability intraday
support/resistance, volume filters breakout strength, 12h EMA ensures alignment with higher timeframe trend.
Works in bull markets (buying strength) and bear markets (selling weakness).
"""

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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla pivot levels for R1, S1, and midpoint (close)
    # Based on previous 4h bar's OHLC
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1 = pivot + (range_4h * 1.1 / 12)
    s1 = pivot - (range_4h * 1.1 / 12)
    midpoint = close_4h  # Camarilla midpoint is the close
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA50
    close_12h_series = pd.Series(close_12h)
    ema_34 = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_50 = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h Camarilla levels to 4h timeframe (no alignment needed)
    r1_aligned = r1
    s1_aligned = s1
    midpoint_aligned = midpoint
    
    # Align 12h EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        midpoint_val = midpoint_aligned[i]
        ema_34_val = ema_34_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.3x avg AND 12h EMA34 > EMA50 (uptrend)
            if price > r1_val and vol > 1.3 * vol_ma and ema_34_val > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.3x avg AND 12h EMA34 < EMA50 (downtrend)
            elif price < s1_val and vol > 1.3 * vol_ma and ema_34_val < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < midpoint OR 12h EMA34 < EMA50 (trend flip)
            if price < midpoint_val or ema_34_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > midpoint OR 12h EMA34 > EMA50 (trend flip)
            if price > midpoint_val or ema_34_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA_Filter"
timeframe = "4h"
leverage = 1.0