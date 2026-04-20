#!/usr/bin/env python3
# 1d_WeeklyPivot_MeanReversion_Volume_Filter
# Hypothesis: Price tends to revert to weekly pivot points (PP) with volume confirmation.
# In ranging markets (2025-2026), mean reversion around pivots works well.
# Uses weekly pivot (PP) and support/resistance levels (S1, R1) from higher timeframe.
# Enters long when price touches S1 with volume > 1.5x average, short when touches R1.
# Exits when price returns to weekly PP. Works in both bull and bear via mean reversion logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_MeanReversion_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using typical price: (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly range
    range_1w = high_1w - low_1w
    # Support and resistance levels
    s1_1w = pp_1w - (range_1w * 0.382)  # 38.2% retracement as S1
    r1_1w = pp_1w + (range_1w * 0.382)  # 38.2% retracement as R1
    
    # Align weekly levels to daily
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Daily volume ratio (current vs 20-day average)
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        pp_val = pp_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(s1_val) or np.isnan(r1_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below S1 with volume confirmation
            if close_val <= s1_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above R1 with volume confirmation
            elif close_val >= r1_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or above weekly pivot (PP)
            if close_val >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or below weekly pivot (PP)
            if close_val <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals