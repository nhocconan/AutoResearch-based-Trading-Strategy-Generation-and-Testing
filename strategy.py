#!/usr/bin/env python3
"""
4h_WeeklyPivot_Trend_Filter_v2
Hypothesis: Weekly pivot levels (R1, S1) act as dynamic support/resistance. 
In trending markets (ADX > 25), price tends to respect these levels. 
In ranging markets (ADX < 20), reversals at R1/S1 offer mean-reversion opportunities.
Combined with volume confirmation to avoid false breakouts.
Designed for low trade frequency (<400 total 4h trades) to minimize fee drag.
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
    
    # Weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align to 4h timeframe (weekly levels only update at weekly close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # ADX for trend strength (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    for i in range(n):
        if i < 1:
            atr[i] = 0
            plus_dm_smooth[i] = 0
            minus_dm_smooth[i] = 0
        elif i < 14:
            atr[i] = (atr[i-1] * (i-1) + tr[i]) / i if i > 0 else tr[i]
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (i-1) + plus_dm[i]) / i if i > 0 else plus_dm[i]
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (i-1) + minus_dm[i]) / i if i > 0 else minus_dm[i]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.zeros(n)
    for i in range(n):
        if i < 14:
            adx[i] = 0
        elif i == 14:
            adx[i] = np.mean(dx[1:15])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price near S1 with volume spike in uptrend (ADX > 25 and rising)
            # OR price breaks above R1 with volume spike in any trend
            near_s1 = abs(close[i] - s1_1w_aligned[i]) / close[i] < 0.005  # Within 0.5%
            above_r1 = close[i] > r1_1w_aligned[i]
            trending_up = adx[i] > 25 and plus_di[i] > minus_di[i]
            
            if (near_s1 and vol_spike[i] and trending_up) or (above_r1 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price near R1 with volume spike in downtrend
            # OR price breaks below S1 with volume spike in any trend
            near_r1 = abs(close[i] - r1_1w_aligned[i]) / close[i] < 0.005  # Within 0.5%
            below_s1 = close[i] < s1_1w_aligned[i]
            trending_down = adx[i] > 25 and minus_di[i] > plus_di[i]
            
            if (near_r1 and vol_spike[i] and trending_down) or (below_s1 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot or trend weakens
            if close[i] < pivot_1w_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot or trend weakens
            if close[i] > pivot_1w_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyPivot_Trend_Filter_v2"
timeframe = "4h"
leverage = 1.0