#!/usr/bin/env python3
"""
6h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: On 6h timeframe, buy when price breaks above daily Camarilla R1 with volume spike (>1.5x median volume) and ATR expansion (ATR > 1.2x median ATR), sell when breaks below daily S1. Uses volume and volatility filters to avoid false breakouts. Target: 15-25 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    return smooth_wilder(tr, period)

def calculate_camarilla(high, low, close):
    # Camarilla pivot levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    range_hl = high - low
    r1 = close + range_hl * 1.1 / 12
    s1 = close - range_hl * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Data (HTF for Camarilla levels, volume, ATR) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR (14-period)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily median ATR (50-period) for expansion filter
    atr_median_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).median().values
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    # Daily Camarilla levels (R1, S1)
    r1_1d, s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily median volume (50-period) for volume spike filter
    vol_median_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_median_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_median_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume and ATR for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        atr_1d_current = atr_1d_aligned[i]
        
        # Volume spike: current volume > 1.5x median volume
        vol_spike = vol_1d_current > 1.5 * vol_median_1d_aligned[i]
        
        # ATR expansion: current ATR > 1.2x median ATR
        atr_expansion = atr_1d_current > 1.2 * atr_median_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above daily R1 with volume spike and ATR expansion
            if close[i] > r1_1d_aligned[i] and vol_spike and atr_expansion:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below daily S1 with volume spike and ATR expansion
            elif close[i] < s1_1d_aligned[i] and vol_spike and atr_expansion:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below S1 (opposite breakout)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above R1 (opposite breakout)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0