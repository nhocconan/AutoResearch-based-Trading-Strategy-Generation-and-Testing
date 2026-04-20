#!/usr/bin/env python3
# 4h_1d_Pivot_R1S1_Breakout_Volume_Confirmation
# Hypothesis: Breakout above 1d Pivot R1 (resistance) or below 1d Pivot S1 (support) with volume confirmation on 4h timeframe.
# Uses 1d trend filter to avoid counter-trend trades. Target: 80-160 trades over 4 years (20-40/year).
# Works in bull/bear via 1d trend filter - only trade with the 1d trend (EMA50).
# Pivot R1 = 2*P - Low, S1 = 2*P - High where P = (H+L+C)/3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R1S1_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate 1d pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla-inspired R1 and S1 levels (more sensitive than R3/S3)
    r1_1d = 2 * pivot_1d - low_1d  # R1 = 2*P - Low
    s1_1d = 2 * pivot_1d - high_1d  # S1 = 2*P - High
    
    # === 1d trend filter: EMA50 ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d levels to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_1d_val = r1_1d_aligned[i]
        s1_1d_val = s1_1d_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_1d_val) or np.isnan(s1_1d_val) or 
            np.isnan(ema50_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with volume confirmation and uptrend
            if (close_val > r1_1d_val and  # Breakout above R1
                ema50_1d_val < close_val and  # Uptrend filter (price above EMA50)
                vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume confirmation and downtrend
            elif (close_val < s1_1d_val and  # Breakdown below S1
                  ema50_1d_val > close_val and  # Downtrend filter (price below EMA50)
                  vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls back below pivot or shows weakness
            if close_val < pivot_1d[i] if not np.isnan(pivot_1d[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above pivot or shows weakness
            if close_val > pivot_1d[i] if not np.isnan(pivot_1d[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals