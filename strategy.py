#!/usr/bin/env python3
"""
6h_Pivot_S1S3_Fade_Trend
Hypothesis: On 6h timeframe, fade price moves to daily S1 (support) with long entries and daily R3 (resistance) with short entries, using 1d EMA34 as trend filter. In strong trends (price beyond S3/R3), continue in trend direction. Uses volume confirmation to avoid false signals. Designed for low trade frequency (~15-25/year) with edge in ranging markets (mean reversion at S1/R3) and strong trends (breakout continuation beyond S3/R3). Works in both bull and bear markets by adapting to regime via EMA filter.
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
    
    # Get 1D data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Pivot Points (standard formula)
    # P = (H + L + C) / 3
    # S1 = (2*P) - H
    # R1 = (2*P) - L
    # S2 = P - (H - L)
    # R2 = P + (H - L)
    # S3 = H - 2*(H - P)
    # R3 = L + 2*(P - L)
    
    # Ensure we have enough data
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivots for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = (2 * pivot) - high_1d
    r1 = (2 * pivot) - low_1d
    s2 = pivot - (high_1d - low_1d)
    r2 = pivot + (high_1d - low_1d)
    s3 = high_1d - 2 * (high_1d - pivot)
    r3 = low_1d + 2 * (pivot - low_1d)
    
    # Align to 6h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # EMA34 on daily close for trend filter
    if len(df_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).values
        ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    else:
        ema34_aligned = np.full(n, np.nan)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        s1 = s1_aligned[i]
        r3 = r3_aligned[i]
        ema34 = ema34_aligned[i]
        
        if position == 0:
            # Long conditions: price at or below S1 (support) in uptrend OR strong break above R3
            if price <= s1 and ema34 > s1 and vol_spike[i]:
                # Mean reversion long at support in uptrend
                signals[i] = 0.25
                position = 1
            elif price >= r3 and ema34 > r3 and vol_spike[i]:
                # Breakout continuation long above R3 in strong uptrend
                signals[i] = 0.25
                position = 1
            # Short conditions: price at or above R3 (resistance) in downtrend OR strong break below S1
            elif price >= r3 and ema34 < r3 and vol_spike[i]:
                # Mean reversion short at resistance in downtrend
                signals[i] = -0.25
                position = -1
            elif price <= s1 and ema34 < s1 and vol_spike[i]:
                # Breakout continuation short below S1 in strong downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses EMA34 or reaches opposite extreme
            if price < ema34 or price >= r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses EMA34 or reaches opposite extreme
            if price > ema34 or price <= s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_S1S3_Fade_Trend"
timeframe = "6h"
leverage = 1.0