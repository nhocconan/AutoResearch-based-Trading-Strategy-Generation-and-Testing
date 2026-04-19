#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot level touch (R1/S1) with 1d volume confirmation and chop filter.
# Long when price touches S1 support with volume spike (>1.8x average) and chop > 61.8 (range).
# Short when price touches R1 resistance with volume spike and chop > 61.8.
# Uses 1d Camarilla levels from prior day (no look-ahead) and 1d chop regime filter.
# Volume confirmation ensures institutional participation at key levels.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).
name = "12h_Camarilla_R1S1_Touch_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla levels (H, L, C from prior day)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, len(high_1d)):
        H = high_1d[i-1]  # Previous day high
        L = low_1d[i-1]   # Previous day low
        C = close_1d[i-1] # Previous day close
        camarilla_r1[i] = C + (H - L) * 1.1 / 12
        camarilla_s1[i] = C - (H - L) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for 1d close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d chopiness index (EHLERS) - needs high/low/close
    # Chop = 100 * log10(sum(ATR1)/ (n * (max(high)-min(low)))) / log10(n)
    # Simplified: use rolling ATR and range
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # Align with original index
        
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        max_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        range_max_min = max_high - min_low
        
        chop = np.full_like(close_arr, np.nan)
        mask = (atr > 0) & (range_max_min > 0) & ~np.isnan(atr) & ~np.isnan(range_max_min)
        chop[mask] = 100 * np.log10(np.sum(atr[mask]) / (window * range_max_min[mask])) / np.log10(window)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current 12h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA and Camarilla data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        chop = chop_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        # Chop filter: range-bound market (chop > 61.8)
        chop_filter = chop > 61.8
        
        if position == 0:
            # Enter long: price touches S1 support with volume and chop confirmation
            if abs(price - s1) < 0.001 * s1 and volume_confirmed and chop_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 resistance with volume and chop confirmation
            elif abs(price - r1) < 0.001 * r1 and volume_confirmed and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price moves above midpoint between S1 and R1 or touches R1
            midpoint = (s1 + r1) / 2
            if price > midpoint or abs(price - r1) < 0.001 * r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price moves below midpoint or touches S1
            midpoint = (s1 + r1) / 2
            if price < midpoint or abs(price - s1) < 0.001 * s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals