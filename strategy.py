#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Donchian breakouts capture sustained moves in both bull and bear markets.
# 1d ATR regime filter avoids whipsaws in low volatility (range) markets.
# Volume confirmation ensures breakouts have conviction.
# Target: 20-40 trades/year with discrete sizing to minimize fee drag.

name = "4h_Donchian20_1dATR_Regime_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_14_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = atr_14_1d / atr_ma_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Donchian(20) channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (1.8x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when volatility is expanding (ATR ratio > 1.0)
        volatile_regime = atr_ratio_1d_aligned[i] > 1.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close breaks above Donchian upper + volatile regime + volume spike
            if close[i] > highest_high[i] and close[i-1] <= highest_high[i] and volatile_regime and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Close breaks below Donchian lower + volatile regime + volume spike
            elif close[i] < lowest_low[i] and close[i-1] >= lowest_low[i] and volatile_regime and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close breaks below Donchian lower or volatility contraction
            if close[i] < lowest_low[i] and close[i-1] >= lowest_low[i] or atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close breaks above Donchian upper or volatility contraction
            if close[i] > highest_high[i] and close[i-1] <= highest_high[i] or atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals