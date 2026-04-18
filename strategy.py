#!/usr/bin/env python3
"""
1d_WeeklyPivotBreakout_TrendAndVolume
Hypothesis: Breakout above/below weekly pivot R1/S1 with 1d volume spike and daily ADX>25 confirms strong momentum. 
Exit when price crosses back below/above S1/R1 or ADX weakens (<20). Designed for low trade frequency (<25/year) 
to avoid fee drag while capturing strong trending moves in both bull and bear markets using higher timeframe (weekly) 
structure and daily confirmation.
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
    
    # Weekly high, low, close for pivot calculation (resistance/support levels)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points and support/resistance levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = pivot_1w + (high_1w - low_1w)  # R1 = P + (H - L)
    s1_1w = pivot_1w - (high_1w - low_1w)  # S1 = P - (H - L)
    
    # Align weekly pivot levels to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily ADX(14) trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20, 14*2)  # Need warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: price > Weekly R1 with volume spike and strong trend (ADX>25)
            if price > r1 and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Weekly S1 with volume spike and strong trend (ADX>25)
            elif price < s1 and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < Weekly S1 OR ADX weakens (<20)
            if price < s1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > Weekly R1 OR ADX weakens (<20)
            if price > r1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivotBreakout_TrendAndVolume"
timeframe = "1d"
leverage = 1.0