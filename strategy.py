#!/usr/bin/env python3
"""
148919: 6h_Pivot_Volume_Regime_Adaptive
Hypothesis: Combines daily pivot points with volatility regime filtering. In low volatility
(regime), fades at pivot extremes (S1/R1); in high volatility, breaks out at S2/R2.
Uses 12h ATR regime filter to avoid whipsaws. Designed for 20-40 trades/year with
adaptive logic that works in both trending and ranging markets.
"""
name = "6h_Pivot_Volume_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily pivot points (standard calculation)
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for pivot calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Standard pivot levels: S1, R1, S2, R2
    S1 = (2 * pivot) - prev_high
    R1 = (2 * pivot) - prev_low
    S2 = pivot - range_hl
    R2 = pivot + range_hl
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    
    # 12h ATR for volatility regime (regime filter)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) on 12h
    atr_12h = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        atr_12h[i] = np.mean(tr[i-13:i+1])
    
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # ATR regime: high vol when current ATR > 1.3 * 50-period average
    atr_ma_50 = np.full_like(atr_12h_aligned, np.nan)
    for i in range(49, len(atr_12h_aligned)):
        atr_ma_50[i] = np.mean(atr_12h_aligned[i-49:i+1])
    
    high_vol_regime = atr_12h_aligned > (1.3 * atr_ma_50)
    
    # Volume confirmation: 20-period volume average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(S2_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(atr_12h_aligned[i]) or
            np.isnan(atr_ma_50[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Determine regime: fade in low vol, breakout in high vol
            if high_vol_regime[i]:
                # High volatility regime: breakout at S2/R2
                if close[i] < S2_aligned[i] and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                elif close[i] > R2_aligned[i] and vol_confirm:
                    signals[i] = 0.25
                    position = 1
            else:
                # Low volatility regime: fade at S1/R1
                if close[i] > R1_aligned[i] and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                elif close[i] < S1_aligned[i] and vol_confirm:
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long exit: reverse signal or pivot crossover
            if high_vol_regime[i]:
                # In high vol, exit on break of S2
                if close[i] < S2_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In low vol, exit on reverse fade signal
                if close[i] < S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or pivot crossover
            if high_vol_regime[i]:
                # In high vol, exit on break of R2
                if close[i] > R2_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In low vol, exit on reverse fade signal
                if close[i] > R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals