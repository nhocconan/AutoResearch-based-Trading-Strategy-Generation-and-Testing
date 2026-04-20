#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with Volume Confirmation
# - Long when price touches weekly S1 pivot with volume spike
# - Short when price touches weekly R1 pivot with volume spike
# - Exit when price reaches weekly pivot point (mean reversion)
# - Uses weekly pivot points for key support/resistance levels
# - Volume filter ensures institutional interest at pivot touches
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot point calculation
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot Point = (High + Low + Close) / 3
    pp = (high_w + low_w + close_w) / 3
    # Resistance 1 = (2 * PP) - Low
    r1 = (2 * pp) - low_w
    # Support 1 = (2 * PP) - High
    s1 = (2 * pp) - high_w
    
    # Align weekly pivot levels to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_w, pp)
    r1_6h = align_htf_to_ltf(prices, df_w, r1)
    s1_6h = align_htf_to_ltf(prices, df_w, s1)
    
    # Volume spike detection on 6h
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 1.5  # 50% above average volume
    
    close = prices['close'].values
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in pivot levels
        if np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price at S1 with volume spike
            if close[i] <= s1_6h[i] * 1.005 and vol_spike[i]:  # within 0.5% of S1
                signals[i] = 0.25
                position = 1
            # Short entry: price at R1 with volume spike
            elif close[i] >= r1_6h[i] * 0.995 and vol_spike[i]:  # within 0.5% of R1
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot point
            if close[i] >= pp_6h[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot point
            if close[i] <= pp_6h[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Reversal_Volume"
timeframe = "6h"
leverage = 1.0