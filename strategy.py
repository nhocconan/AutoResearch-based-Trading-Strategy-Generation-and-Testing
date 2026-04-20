#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    r2 = close_1d + (range_hl * 1.1 / 6)
    s2 = close_1d - (range_hl * 1.1 / 6)
    r3 = close_1d + (range_hl * 1.1 / 4)
    s3 = close_1d - (range_hl * 1.1 / 4)
    r4 = close_1d + (range_hl * 1.1 / 2)
    s4 = close_1d - (range_hl * 1.1 / 2)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === Daily ATR-based volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_series = pd.Series(tr)
    atr_14 = atr_series.rolling(window=14, min_periods=14).mean().values
    atr_ma30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    vol_ratio = atr_14 / np.where(atr_ma30 > 0, atr_ma30, np.nan)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio with proper initialization
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio_current = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio_aligned[i]
        vol_ratio_current_val = vol_ratio_current[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(vol_ratio_current_val) or 
            np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(r4_val) or np.isnan(s4_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with low volatility and volume confirmation
            if (close_val > r1_val and 
                vol_ratio_val < 0.8 and  # Low volatility contraction
                vol_ratio_current_val > 1.8):  # Volume expansion
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with low volatility and volume confirmation
            elif (close_val < s1_val and 
                  vol_ratio_val < 0.8 and  # Low volatility contraction
                  vol_ratio_current_val > 1.8):  # Volume expansion
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to S1 or volatility expands too much
            if close_val < s1_val or vol_ratio_val > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to R1 or volatility expands too much
            if close_val > r1_val or vol_ratio_val > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals