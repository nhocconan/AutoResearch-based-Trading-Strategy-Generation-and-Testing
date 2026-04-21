#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Regime
Hypothesis: 12h Camarilla pivot breakout with volume confirmation and chop regime filter.
Works in bull/bear by using pivot levels as support/resistance and chop filter to avoid whipsaw.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels (R1, S1)"""
    pivot = (high + low + close) / 3
    range_hl = high - low
    r1 = close + range_hl * 1.1 / 12
    s1 = close - range_hl * 1.1 / 12
    return pivot, r1, s1

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    atr = np.zeros(len(high))
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    sum_tr = np.zeros(len(high))
    for i in range(period-1, len(high)):
        sum_tr[i] = np.sum(tr[i-period+1:i+1])
    
    chop = np.zeros(len(high))
    for i in range(period-1, len(high)):
        chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * period)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot and chop regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1, S1 on daily data
    pivot_1d = np.full_like(high_1d, np.nan)
    r1_1d = np.full_like(high_1d, np.nan)
    s1_1d = np.full_like(high_1d, np.nan)
    
    for i in range(len(high_1d)):
        _, r1, s1 = calculate_camarilla_pivot(high_1d[i], low_1d[i], close_1d[i])
        pivot_1d[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Calculate Chop on daily data
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Chop regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending
        chop_val = chop_1d_aligned[i]
        is_range = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above R1 in range regime with volume
            if price > r1_1d_aligned[i] and is_range and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in range regime with volume
            elif price < s1_1d_aligned[i] and is_range and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below pivot or chop breaks down (trending)
            if price < pivot_1d_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot or chop breaks down (trending)
            if price > pivot_1d_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0