#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Camarilla Pivot Breakout with Daily Volume Spike and ATR Filter
- Uses 1w Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
- Daily ATR(14) filter: only trade when ATR > 0.5 * ATR(50) to avoid low-volatility chop
- Daily volume confirmation: volume > 1.5 * 20-day average for institutional participation
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by combining mean reversion (R3/S3) and breakout (R4/S4) logic
- Weekly HTF provides stable structure, daily filters add precision
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla calculations
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    R3_1w = pivot_1w + range_1w * 1.1 / 2
    S3_1w = pivot_1w - range_1w * 1.1 / 2
    R4_1w = pivot_1w + range_1w * 1.1
    S4_1w = pivot_1w - range_1w * 1.1
    
    # Align HTF levels to LTF
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Daily volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for ATR50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_1w_aligned[i]) or np.isnan(S3_1w_aligned[i]) or 
            np.isnan(R4_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR(14) > 0.5 * ATR(50)
        vol_filter = atr_14_aligned[i] > 0.5 * atr_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long conditions:
            # 1. R3 reversal: price crosses above S3 with volume and volatility
            # 2. R4 breakout: price breaks above R4 with volume and volatility
            long_reversal = (close[i] > S3_1w_aligned[i] and 
                           close[i-1] <= S3_1w_aligned[i-1] and
                           vol_filter and vol_confirm)
            long_breakout = (close[i] > R4_1w_aligned[i] and 
                           vol_filter and vol_confirm)
            
            # Short conditions:
            # 1. S3 reversal: price crosses below R3 with volume and volatility
            # 2. S4 breakdown: price breaks below S4 with volume and volatility
            short_reversal = (close[i] < R3_1w_aligned[i] and 
                            close[i-1] >= R3_1w_aligned[i-1] and
                            vol_filter and vol_confirm)
            short_breakout = (close[i] < S4_1w_aligned[i] and 
                            vol_filter and vol_confirm)
            
            if long_reversal or long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_reversal or short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below S3 (reversal fail) or crosses above R4 (take profit)
                if (close[i] < S3_1w_aligned[i] and close[i-1] >= S3_1w_aligned[i-1]) or \
                   (close[i] > R4_1w_aligned[i] and close[i-1] <= R4_1w_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above R3 (reversal fail) or crosses below S4 (take profit)
                if (close[i] > R3_1w_aligned[i] and close[i-1] <= R3_1w_aligned[i-1]) or \
                   (close[i] < S4_1w_aligned[i] and close[i-1] >= S4_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyCamarilla_R3S4_Breakout_1dATR_VolumeFilter"
timeframe = "6h"
leverage = 1.0