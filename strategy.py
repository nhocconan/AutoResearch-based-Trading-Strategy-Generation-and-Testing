#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Volume_Trend
Hypothesis: On 12h timeframe, price approaching Camarilla pivot levels (R3/S3) with volume confirmation and weekly trend filter provides high-probability mean-reversion entries in ranging markets and trend-following breakouts in trending markets. Weekly trend from EMA34 on 1w filters direction. Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drift while capturing reversals and breakouts in both bull and bear regimes.
"""

name = "12h_Camarilla_Pivot_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3, R2, R1, PP, S1, S2, S3
    # R3 = Close + 1.1 * (High - Low) * 1.1/2? Actually standard: R3 = Close + 1.1*(High-Low)
    # Standard Camarilla:
    # R4 = Close + ((High-Low)*1.1/2)
    # R3 = Close + ((High-Low)*1.1/4)
    # R2 = Close + ((High-Low)*1.1/6)
    # R1 = Close + ((High-Low)*1.1/12)
    # PP = (High+Low+Close)/3
    # S1 = Close - ((High-Low)*1.1/12)
    # S2 = Close - ((High-Low)*1.1/6)
    # S3 = Close - ((High-Low)*1.1/4)
    # S4 = Close - ((High-Low)*1.1/2)
    # We use R3 and S3 for fade, R4/S4 for breakout
    diff = high_1d - low_1d
    r3 = close_1d + 1.1 * diff * 0.5  # Actually 1.1/2 = 0.5
    s3 = close_1d - 1.1 * diff * 0.5
    r4 = close_1d + 1.1 * diff * 0.5  # Same as R3? Wait, recalc: R4 = Close + 1.1*(High-Low)*1/2? Let's use standard multipliers
    # Correct Camarilla multipliers:
    # R4 = Close + (High-Low) * 1.1/2
    # R3 = Close + (High-Low) * 1.1/4
    # R2 = Close + (High-Low) * 1.1/6
    # R1 = Close + (High-Low) * 1.1/12
    # S1 = Close - (High-Low) * 1.1/12
    # S2 = Close - (High-Low) * 1.1/6
    # S3 = Close - (High-Low) * 1.1/4
    # S4 = Close - (High-Low) * 1.1/2
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly trend: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (need prior day) and weekly EMA34
    start_idx = 1  # Camarilla uses prior day, so need at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly EMA34
        uptrend_w = close[i] > ema34_1w_aligned[i]
        downtrend_w = close[i] < ema34_1w_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Price proximity to Camarilla levels (within 0.1%)
        prox_r3 = abs(close[i] - r3_aligned[i]) / close[i] < 0.001
        prox_r4 = abs(close[i] - r4_aligned[i]) / close[i] < 0.001
        prox_s3 = abs(close[i] - s3_aligned[i]) / close[i] < 0.001
        prox_s4 = abs(close[i] - s4_aligned[i]) / close[i] < 0.001
        
        if position == 0:
            # Long setup: near S3/S4 in uptrend (mean reversion) OR break above R4 in uptrend (breakout)
            # Short setup: near R3/R4 in downtrend (mean reversion) OR break below S4 in downtrend (breakdown)
            if uptrend_w:
                # Long: mean reversion off S3/S4 or breakout above R4
                if (prox_s3 or prox_s4) and volume_filter:
                    signals[i] = 0.25
                    position = 1
                elif prox_r4 and volume_filter:  # Breakout above R4
                    signals[i] = 0.25
                    position = 1
            elif downtrend_w:
                # Short: mean reversion off R3/R4 or breakdown below S4
                if (prox_r3 or prox_r4) and volume_filter:
                    signals[i] = -0.25
                    position = -1
                elif prox_s4 and volume_filter:  # Breakdown below S4
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: failed mean reversion (goes to S4) or trend fails
            if prox_s4 or not uptrend_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: failed mean reversion (goes to R4) or trend fails
            if prox_r4 or not downtrend_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals