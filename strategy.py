#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_RegimeFilter_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and choppiness regime filter. 
Only trade breakouts aligned with weekly trend in non-choppy markets. Uses discrete position sizing (0.25) to minimize fee drag.
Target: 15-30 trades/year per symbol (~60-120 total over 4 years) to avoid fee drag.
Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
Choppiness filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla levels and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla width
    rang = high_1d - low_1d
    
    # Resistance levels R1, R3
    r1 = close_1d_prev + rang * 1.1 / 12
    r3 = close_1d_prev + rang * 1.1 / 4
    
    # Support levels S1, S3
    s1 = close_1d_prev - rang * 1.1 / 12
    s3 = close_1d_prev - rang * 1.1 / 4
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate daily choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (log10(highest_high - lowest_low) * 14))
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr = np.where(np.arange(len(tr1)) == 0, tr1, tr2)  # First bar TR = high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and log of zero
    hh_ll_diff = highest_high_14 - lowest_low_14
    chop_raw = np.where((hh_ll_diff > 0) & (atr14 > 0), 
                        np.sum([atr14[i-13:i+1] for i in range(13, len(atr14))], axis=1) if len(atr14) >= 14 else np.zeros_like(atr14),
                        100)  # Default to choppy when invalid
    
    # Simplified chop calculation: 100 * log10(sum(ATR14) / log10(hh_ll_diff) / 14)
    sum_atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr14 / np.log10(hh_ll_diff) / 14)
    chop = np.where((hh_ll_diff > 1) & (sum_atr14 > 0), chop, 50)  # Default to middle when invalid
    
    # Align chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detector (20-day volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Regime filter: avoid choppy markets (CHOP > 61.8 = ranging)
        not_choppy = chop_aligned[i] <= 61.8
        
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
            if close[i] < r1_aligned[i] or not uptrend or not not_choppy:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S1 OR trend changes OR market becomes choppy
            if close[i] > s1_aligned[i] or not downtrend or not not_choppy:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0