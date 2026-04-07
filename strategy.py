#!/usr/bin/env python3
"""
1d_volatility_breakout_v1
Hypothesis: Uses 1-day Donchian breakout with weekly volatility filter to capture breakouts in both bull and bear markets. Long when price breaks above 20-day high with expanding volatility, short when breaks below 20-day low with expanding volatility. Weekly ATR filter ensures we only trade when volatility is increasing, avoiding choppy markets. Position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volatility_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-day Donchian channels
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Weekly ATR for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation for weekly
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) on weekly
    atr_period = 14
    atr_1w = np.zeros_like(tr_1w)
    atr_1w[atr_period-1] = np.mean(tr_1w[:atr_period])
    for i in range(atr_period, len(tr_1w)):
        atr_1w[i] = (atr_1w[i-1] * (atr_period - 1) + tr_1w[i]) / atr_period
    
    # ATR ratio: current ATR / ATR 4 periods ago (to detect expansion)
    atr_ratio = np.zeros_like(atr_1w)
    for i in range(4, len(atr_1w)):
        if atr_1w[i-4] > 0:
            atr_ratio[i] = atr_1w[i] / atr_1w[i-4]
        else:
            atr_ratio[i] = 1.0
    
    # Align ATR ratio to daily
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback*2, n):  # Start after warmup
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion filter: ATR ratio > 1.1 (10% expansion)
        vol_expansion = atr_ratio_aligned[i] > 1.1
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or volatility contracts
            if close[i] < donchian_low[i] or atr_ratio_aligned[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or volatility contracts
            if close[i] > donchian_high[i] or atr_ratio_aligned[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volatility expansion
            if close[i] > donchian_high[i] and vol_expansion:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volatility expansion
            elif close[i] < donchian_low[i] and vol_expansion:
                position = -1
                signals[i] = -0.25
    
    return signals