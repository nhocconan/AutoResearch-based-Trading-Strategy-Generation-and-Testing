#!/usr/bin/env python3
name = "4h_Choppiness_Index_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Choppiness Index for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods for 1d
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    sum_tr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_1d / (hh_1d - ll_1d)) / np.log10(14)
    chop[hh_1d == ll_1d] = 50  # Avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Donchian Channel (20) for breakout ===
    period_dc = 20
    upper_dc = pd.Series(high).rolling(window=period_dc, min_periods=period_dc).max().values
    lower_dc = pd.Series(low).rolling(window=period_dc, min_periods=period_dc).min().values
    
    # === Volume spike detection (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(upper_dc[i]) or
            np.isnan(lower_dc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        chop_val = chop_1d_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + chop > 61.8 (range) + volume spike
            if (close[i] > upper_dc[i] and 
                chop_val > 61.8 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + chop > 61.8 (range) + volume spike
            elif (close[i] < lower_dc[i] and 
                  chop_val > 61.8 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakout down or chop < 38.2 (trend)
            if close[i] < lower_dc[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout up or chop < 38.2 (trend)
            if close[i] > upper_dc[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals