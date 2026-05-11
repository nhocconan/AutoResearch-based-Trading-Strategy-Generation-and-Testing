#!/usr/bin/env python3
"""
6h_Weekly_Camarilla_R4S4_Breakout_Trend_v1
Hypothesis: Uses weekly Camarilla pivot levels (R4/S4) for breakout trading with 1d trend filter.
In strong trends, price tends to break beyond R4/S4 levels. We enter on breakouts with volume
confirmation only when aligned with the 1d EMA50 trend. Weekly pivots provide structural levels
that work in both bull and bear markets as they adapt to volatility. Designed for low trade
frequency by requiring both weekly level breakout and volume confirmation.
"""

name = "6h_Weekly_Camarilla_R4S4_Breakout_Trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Camarilla Pivot Levels (R4, S4) ---
    # Calculate from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range for weekly
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Pivot point (standard)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Camarilla levels
    r4_1w = pivot_1w + (high_1w - low_1w) * 1.1 / 2
    s4_1w = pivot_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align weekly levels to 6h
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # --- Volume Spike Detection (24-period average) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- 1d Trend Filter (EMA50) ---
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume, above 1d EMA50
            if (close[i] > r4_1w_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 with volume, below 1d EMA50
            elif (close[i] < s4_1w_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to weekly pivot area or loss of momentum
            if position == 1:
                # Exit long: price returns below weekly pivot
                if close[i] < pivot_1w[i] if not np.isnan(pivot_1w[i]) else pivot_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns above weekly pivot
                if close[i] > pivot_1w[i] if not np.isnan(pivot_1w[i]) else pivot_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals