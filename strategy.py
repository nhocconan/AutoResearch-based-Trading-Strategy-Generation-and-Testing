#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v5
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and choppiness regime filter. Only trade breakouts aligned with daily trend in non-choppy markets. Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year per symbol (~50-150 total over 4 years) to avoid fee drag. Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
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
    
    # Get 1d data for Camarilla levels, EMA50 trend, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR14
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(H-L) over 14 periods
    max_hl_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).max().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / (atr14 * 14)) / log10(14)
    chop_denom = atr_14 * 14
    chop_ratio = np.where((chop_denom > 0) & (max_hl_14 > 0), sum_tr_14 / max_hl_14, np.nan)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d_prev = df_1d['high'].values
    low_1d_prev = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla width
    rang = high_1d_prev - low_1d_prev
    
    # Resistance levels
    r1 = close_1d_prev + rang * 1.1 / 12
    s1 = close_1d_prev - rang * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike detector (20-bar volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Regime filter: only trade when not choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in uptrend and not choppy
            if close[i] > r1_aligned[i] and volume_spike[i] and uptrend and not_choppy:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in downtrend and not choppy
            elif close[i] < s1_aligned[i] and volume_spike[i] and downtrend and not_choppy:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R1 OR trend changes OR market becomes choppy
            if close[i] < r1_aligned[i] or not uptrend or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S1 OR trend changes OR market becomes choppy
            if close[i] > s1_aligned[i] or not downtrend or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v5"
timeframe = "12h"
leverage = 1.0