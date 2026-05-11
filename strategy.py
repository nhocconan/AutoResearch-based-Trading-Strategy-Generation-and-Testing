#!/usr/bin/env python3
"""
12h_PivotPoint_Bounce_1dATR_Filter
Hypothesis: Trades reversals at daily pivot points (S1/S2 for longs, R1/R2 for shorts) with 12h confirmation.
Uses price rejection at key levels with ATR-based volatility filter to avoid chop. Works in both bull and bear markets
by fading extremes at institutional support/resistance. Low trade frequency expected (<30/year) to minimize fee drag.
"""

name = "12h_PivotPoint_Bounce_1dATR_Filter"
timeframe = "12h"
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
    
    # === 1D Data for Pivot Points and ATR ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    s1_1d = 2 * pp_1d - high_1d
    s2_1d = pp_1d - (high_1d - low_1d)
    r1_1d = 2 * pp_1d - low_1d
    r2_1d = pp_1d + (high_1d - low_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Align pivot levels to 12h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers ATR calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period MA (avoid chop)
        atr_ma = pd.Series(atr_14_aligned[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1] if i >= 50 else np.nan
        if np.isnan(atr_ma) or atr_14_aligned[i] < 0.8 * atr_ma:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long near S1/S2: price rejects support with bullish close
            if ((abs(close[i] - s1_aligned[i]) < 0.5 * atr_14_aligned[i] or 
                 abs(close[i] - s2_aligned[i]) < 0.5 * atr_14_aligned[i]) and
                close[i] > open_prices[i]):  # bullish candle
                signals[i] = 0.25
                position = 1
            # Short near R1/R2: price rejects resistance with bearish close
            elif ((abs(close[i] - r1_aligned[i]) < 0.5 * atr_14_aligned[i] or 
                   abs(close[i] - r2_aligned[i]) < 0.5 * atr_14_aligned[i]) and
                  close[i] < open_prices[i]):  # bearish candle
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot point or shows rejection at resistance
            if (close[i] > pp_aligned[i] or 
                (abs(close[i] - r1_aligned[i]) < 0.5 * atr_14_aligned[i] and close[i] < open_prices[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price reaches pivot point or shows rejection at support
            if (close[i] < pp_aligned[i] or 
                (abs(close[i] - s1_aligned[i]) < 0.5 * atr_14_aligned[i] and close[i] > open_prices[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals

# Ensure open_prices is defined
open_prices = prices['open'].values if 'prices' in locals() else np.array([])  # This line will be replaced in actual context
# Fix: move open_prices extraction to top
# Actually, let's restructure properly: