#!/usr/bin/env python3
# 6h_1d_Pivot_R4S4_Breakout_Volume_Spike
# Hypothesis: Breakouts at daily Pivot R4/S3 levels with volume spike and ATR filter.
# Uses 1d trend filter (close vs SMA50) to align with trend direction.
# Works in bull/bear via 1d trend filter - only trade breakouts in direction of 1d trend.
# Target: 60-100 total trades over 4 years (15-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R4S4_Breakout_Volume_Spike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate 1d pivot levels (R4, S4) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4 = close + (range * 1.1/2), S4 = close - (range * 1.1/2)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # === 1d trend filter: close vs SMA50 ===
    close_s = pd.Series(close_1d)
    sma50_1d = close_s.rolling(window=50, min_periods=50).mean().values
    
    # === 6h: Volume spike (current vs 24-period average) ===
    volume = prices['volume'].values
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma24 > 0, vol_ma24, np.nan)
    
    # === 6h: ATR filter (ATR(24) > 1.5 * ATR(96)) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr24 = pd.Series(tr).rolling(window=24, min_periods=24).mean().values
    atr96 = pd.Series(tr).rolling(window=96, min_periods=96).mean().values
    atr_filter = atr24 > (1.5 * atr96)
    
    # Align all 1d levels to 6h
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(96, n):  # Start after ATR96 warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r4_1d_val = r4_1d_aligned[i]
        s4_1d_val = s4_1d_aligned[i]
        sma50_1d_val = sma50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_filter_val = bool(atr_filter_aligned[i])
        
        # Skip if any value is NaN
        if (np.isnan(r4_1d_val) or np.isnan(s4_1d_val) or np.isnan(sma50_1d_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R4 with volume spike, ATR filter, and above 1d SMA50
            if (close_val > r4_1d_val and 
                vol_ratio_val > 2.5 and 
                atr_filter_val and 
                close_val > sma50_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 with volume spike, ATR filter, and below 1d SMA50
            elif (close_val < s4_1d_val and 
                  vol_ratio_val > 2.5 and 
                  atr_filter_val and 
                  close_val < sma50_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below R4 or trend reversal
            if close_val < r4_1d_val or close_val < sma50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above S4 or trend reversal
            if close_val > s4_1d_val or close_val > sma50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals