#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2
Camarilla R1/S1 breakout with volume spike and ATR volatility filter.
Long: break above R1 with volume > 1.5x average and ATR > 0.5x ATR(50)
Short: break below S1 with volume > 1.5x average and ATR > 0.5x ATR(50)
Exit: price re-enters between H3 and L3 or ATR drops below 0.3x ATR(50)
Designed to capture institutional breakouts with volatility confirmation.
Target: 20-50 trades per year (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === ATR(14) and ATR(50) for volatility filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # === Volume average (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily Camarilla levels ===
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for current day using previous day's data
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    h3 = prev_close + camarilla_range * 1.1 / 4
    l3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or np.isnan(atr50[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike condition (1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # ATR filter: current ATR > 0.5 * ATR50 for entry, < 0.3 * ATR50 for exit
        vol_filter_entry = atr[i] > 0.5 * atr50[i]
        vol_filter_exit = atr[i] < 0.3 * atr50[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume spike and sufficient volatility
            if (close[i] > r1_aligned[i] and 
                volume_spike and vol_filter_entry):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume spike and sufficient volatility
            elif (close[i] < s1_aligned[i] and 
                  volume_spike and vol_filter_entry):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: re-entry below H3 OR low volatility
            if (close[i] < h3_aligned[i] or vol_filter_exit):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: re-entry above S3 OR low volatility
            if (close[i] > l3_aligned[i] or vol_filter_exit):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2"
timeframe = "12h"
leverage = 1.0