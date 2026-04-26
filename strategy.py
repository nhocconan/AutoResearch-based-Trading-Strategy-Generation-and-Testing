#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter
Hypothesis: 12h Camarilla R1/S1 breakouts with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar MA). 
Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year). Uses tighter entry conditions to avoid overtrading.
Breakouts of Camarilla R1/S1 levels indicate strong momentum. 1d EMA50 ensures alignment with daily trend.
Volume confirmation filters out weak breakouts. Works in both bull (trend continuation) and bear (mean reversion to midpoint) markets.
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels (R1/S1 = standard breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (standard breakout levels)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_vals + (rng * 1.1 / 12)   # R1 level
    camarilla_s1 = close_1d_vals - (rng * 1.1 / 12)   # S1 level
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Fixed position size to minimize fee churn (0.25 = 25% of capital)
    FIXED_SIZE = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA, 20 for vol)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry conditions: breakout of Camarilla R1/S1 with volume spike AND 1d EMA50 trend filter
        long_entry = (close_val > camarilla_r1_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < camarilla_s1_val) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = FIXED_SIZE
                position = 1
            elif short_entry:
                signals[i] = -FIXED_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = FIXED_SIZE
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -FIXED_SIZE
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0