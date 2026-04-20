#!/usr/bin/env python3
# 12h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter
# Hypothesis: Breakout above R1 or below S1 daily pivot levels with volume confirmation and ATR volatility filter on 12h timeframe.
# Uses 12h close vs 1d pivot levels for entry, volume spike for confirmation, and ATR to filter low-volatility periods.
# Works in bull/bear via breakout logic - only trade when price breaks key levels with momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate 1d pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # === 12h: ATR(14) for volatility filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d data to 12h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after ATR/volume warmup
        # Get values
        close_val = close[i]
        r1_1d_val = r1_1d_aligned[i]
        s1_1d_val = s1_1d_aligned[i]
        atr_val = atr[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_1d_val) or np.isnan(s1_1d_val) or 
            np.isnan(atr_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR > 0 (avoid dead markets)
        vol_filter = atr_val > 0
        
        if position == 0:
            # Long: Breakout above R1 with volume confirmation and volatility
            if (close_val > r1_1d_val and  # Price broke above R1
                vol_filter and
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume confirmation and volatility
            elif (close_val < s1_1d_val and  # Price broke below S1
                  vol_filter and
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below S1 (invalidates breakout)
            if close_val < s1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above R1 (invalidates breakdown)
            if close_val > r1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals