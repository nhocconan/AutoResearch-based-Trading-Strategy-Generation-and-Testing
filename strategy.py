#!/usr/bin/env python3
"""
4h Camarilla Pivot (1d) + Volume Spike + Choppiness Regime (1d) for BTC/ETH.
Long when price closes above H3 with volume > 2x avg and CHOP > 50 (range).
Short when price closes below L3 with volume > 2x avg and CHOP > 50.
Exit when price reaches H4/L4 or CHOP < 40 (trending).
Designed for low turnover: ~20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data once for Camarilla pivot and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev[0] = close_1d[0]
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 6
    camarilla_l3 = close_prev - range_prev * 1.1 / 6
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Calculate Choppiness Index (14)
    chop_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=chop_period, adjust=False, min_periods=chop_period).mean().values
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = np.where(
        (highest_high - lowest_low) > 0,
        100 * np.log10(sum_tr / atr / chop_period) / np.log10(chop_period),
        50
    )
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # 1-day index (6 bars per day for 4h timeframe)
        idx_1d = i // 6
        if idx_1d < 1:
            continue
        
        # Use previous 1d values to avoid look-ahead
        prev_idx = idx_1d - 1
        if prev_idx < 0:
            continue
            
        # Get values from previous 1d bar
        h3_prev = camarilla_h3[prev_idx] if prev_idx < len(camarilla_h3) else camarilla_h3[-1]
        l3_prev = camarilla_l3[prev_idx] if prev_idx < len(camarilla_l3) else camarilla_l3[-1]
        h4_prev = camarilla_h4[prev_idx] if prev_idx < len(camarilla_h4) else camarilla_h4[-1]
        l4_prev = camarilla_l4[prev_idx] if prev_idx < len(camarilla_l4) else camarilla_l4[-1]
        chop_prev = chop[prev_idx] if prev_idx < len(chop) else chop[-1]
        
        if np.isnan(h3_prev) or np.isnan(l3_prev) or np.isnan(h4_prev) or np.isnan(l4_prev) or np.isnan(chop_prev):
            continue
        
        # Create arrays for alignment
        h3_arr = np.full(len(df_1d), h3_prev)
        l3_arr = np.full(len(df_1d), l3_prev)
        h4_arr = np.full(len(df_1d), h4_prev)
        l4_arr = np.full(len(df_1d), l4_prev)
        chop_arr = np.full(len(df_1d), chop_prev)
        
        h3_4h = align_htf_to_ltf(prices, df_1d, h3_arr)[i]
        l3_4h = align_htf_to_ltf(prices, df_1d, l3_arr)[i]
        h4_4h = align_htf_to_ltf(prices, df_1d, h4_arr)[i]
        l4_4h = align_htf_to_ltf(prices, df_1d, l4_arr)[i]
        chop_4h = align_htf_to_ltf(prices, df_1d, chop_arr)[i]
        
        if np.isnan(h3_4h) or np.isnan(l3_4h) or np.isnan(h4_4h) or np.isnan(l4_4h) or np.isnan(chop_4h):
            continue
        
        if position == 0:
            # Long: Close above H3 with volume spike and chop > 50 (range)
            if close[i] > h3_4h and volume[i] > vol_ma[i] * 2.0 and chop_4h > 50:
                position = 1
                signals[i] = position_size
            # Short: Close below L3 with volume spike and chop > 50 (range)
            elif close[i] < l3_4h and volume[i] > vol_ma[i] * 2.0 and chop_4h > 50:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Reach H4 or chop < 40 (trending)
            if close[i] >= h4_4h or chop_4h < 40:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Reach L4 or chop < 40 (trending)
            if close[i] <= l4_4h or chop_4h < 40:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volume_Chop"
timeframe = "4h"
leverage = 1.0