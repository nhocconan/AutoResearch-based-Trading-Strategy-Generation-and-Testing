#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and weekly pivot direction.
Long when price breaks above Donchian upper band AND 1d ATR(14) < 1d ATR(50) (low volatility regime) AND price > weekly pivot.
Short when price breaks below Donchian lower band AND 1d ATR(14) < 1d ATR(50) AND price < weekly pivot.
Exit when price touches the opposite Donchian band or ATR regime shifts to high volatility.
Uses 1d HTF for ATR regime filter and 1w HTF for pivot direction (avoids whipsaws in ranging markets).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate 1w pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 50, 50)  # Donchian (20), ATR(50) (50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr14 = atr_14_aligned[i]
        atr50 = atr_50_aligned[i]
        pivot = pivot_1w_aligned[i]
        up = upper[i]
        lo = lower[i]
        
        # Low volatility regime: ATR(14) < ATR(50)
        low_vol_regime = atr14 < atr50
        
        if position == 0:
            # Long: Break above Donchian upper AND low vol regime AND price > weekly pivot
            if price > up and low_vol_regime and price > pivot:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND low vol regime AND price < weekly pivot
            elif price < lo and low_vol_regime and price < pivot:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower band OR high volatility regime
                if price < lo or not low_vol_regime:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper band OR high volatility regime
                if price > up or not low_vol_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1dATR_LowVol_1wPivot_Filter"
timeframe = "6h"
leverage = 1.0