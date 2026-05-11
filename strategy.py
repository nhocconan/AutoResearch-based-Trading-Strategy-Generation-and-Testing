#!/usr/bin/env python3
"""
6h_Weekly_Pivot_MeanReversion_v1
Hypothesis: Mean reversion at weekly pivot levels (S1/S2/R1/R2) with 1d trend filter.
In ranging markets, price tends to revert from S1/R1 toward pivot. In trending markets,
breakouts through S2/R2 with trend alignment continue. Works in both bull/bear by
adapting to regime via 1d EMA filter. Targets low-frequency, high-conviction trades.
"""
name = "6h_Weekly_Pivot_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Pivot Calculation (using prior week) ---
    # Prior week high, low, close
    wh = df_1w['high'].shift(1).values  # shift to use prior week only
    wl = df_1w['low'].shift(1).values
    wc = df_1w['close'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (wh + wl + wc) / 3.0
    s1 = (2 * pivot) - wh
    s2 = pivot - (wh - wl)
    r1 = (2 * pivot) - wl
    r2 = pivot + (wh - wl)
    
    # Align weekly levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    
    # --- 1d Trend Filter (EMA34) ---
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long setup: price near S1/S2 with bullish alignment
            near_s1 = low[i] <= s1_aligned[i] * 1.002  # within 0.2% of S1
            near_s2 = low[i] <= s2_aligned[i] * 1.002  # within 0.2% of S2
            bullish = close[i] > ema_34_aligned[i]
            
            if (near_s1 or near_s2) and bullish and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # Short setup: price near R1/R2 with bearish alignment
            near_r1 = high[i] >= r1_aligned[i] * 0.998  # within 0.2% of R1
            near_r2 = high[i] >= r2_aligned[i] * 0.998  # within 0.2% of R2
            bearish = close[i] < ema_34_aligned[i]
            
            if (near_r1 or near_r2) and bearish and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches pivot or shows weakness
            if close[i] >= pivot_aligned[i] * 0.999 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot or shows strength
            if close[i] <= pivot_aligned[i] * 1.001 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals