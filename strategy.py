#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_ATRFilter
Hypothesis: Use daily Camarilla pivot levels (R1/S1) on 6h timeframe with volume and ATR confirmation.
Go long when price breaks above R1 with volume > 1.5x average and ATR > ATR(50) * 0.5.
Go short when price breaks below S1 with same filters.
Exit when price returns to the pivot (midpoint) or opposite level is touched.
Designed for 6h to capture multi-day moves while avoiding whipsaw in low volatility.
Works in bull/bear by following breakouts with volatility filter.
Target: 15-25 trades/year (~60-100 total over 4 years).
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # Pivot = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # ATR for volatility filter
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full_like(close, np.nan)
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr[i] = np.mean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Volume average
    vol_ma_period = 20
    vol_ma = np.full_like(volume, np.nan)
    for i in range(vol_ma_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility and volume filters
        vol_filter = atr[i] > np.nanmedian(atr[max(0, i-100):i]) * 0.5 if not np.isnan(np.nanmedian(atr[max(0, i-100):i])) else True
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: break above R1
            if close[i] > r1_1d_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1
            elif close[i] < s1_1d_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: exit if price returns to pivot or breaks S1 (contrarian)
            if close[i] <= pivot_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: exit if price returns to pivot or breaks R1
            if close[i] >= pivot_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0