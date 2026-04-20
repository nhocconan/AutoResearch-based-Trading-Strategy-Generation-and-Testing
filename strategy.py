#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 5 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1_1d = pivot_1d + (range_1d * 1.1 / 12)
    S1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # === 1w: Calculate weekly EMA200 for trend filter ===
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    ema200_12h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        r1_val = R1_12h[i]
        s1_val = S1_12h[i]
        ema200_val = ema200_12h[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema200_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and above weekly EMA200
            if close_val > r1_val and vol_ratio_val > 2.0 and close_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and below weekly EMA200
            elif close_val < s1_val and vol_ratio_val > 2.0 and close_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below R1 or low volume or below weekly EMA200
            if close_val < r1_val or vol_ratio_val < 0.9 or close_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above S1 or low volume or above weekly EMA200
            if close_val > s1_val or vol_ratio_val < 0.9 or close_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals