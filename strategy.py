#!/usr/bin/env python3
"""
6h_12h_Pivot_Reversion
Hypothesis: On 6h chart, mean-revert from 12h pivot (mean of high, low, close) when price deviates >1.5*ATR(12h),
with confirmation from 1d trend (price > EMA50 for longs, < EMA50 for shorts). Works in both bull and bear
markets by fading overextensions relative to the 12h pivot while respecting the daily trend.
"""

name = "6h_12h_Pivot_Reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 12h data for pivot and ATR ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h pivot (mean of high, low, close)
    pivot_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    # 12h ATR(14)
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    # Align 12h pivot and ATR to 6h
    pivot_12h_6h = align_htf_to_ltf(prices, df_12h, pivot_12h.values)
    atr_12h_6h = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # --- 1d EMA50 trend filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_12h_6h[i]) or np.isnan(atr_12h_6h[i]) or 
            np.isnan(ema_50_1d_6h[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        dev = close[i] - pivot_12h_6h[i]  # deviation from pivot
        atr = atr_12h_6h[i]
        
        if position == 0:
            # Long: price below pivot by >1.5*ATR and above 1d EMA50
            if dev < -1.5 * atr and close[i] > ema_50_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above pivot by >1.5*ATR and below 1d EMA50
            elif dev > 1.5 * atr and close[i] < ema_50_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to pivot (within 0.5*ATR) or trend fails
            if position == 1:
                if abs(dev) < 0.5 * atr or close[i] < ema_50_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if abs(dev) < 0.5 * atr or close[i] > ema_50_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals