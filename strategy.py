#!/usr/bin/env python3
"""
6H Ehlers Fisher Transform with Volume Confirmation and 12h Trend Filter
- Uses Ehlers Fisher Transform (period=9) to identify turning points
- Long when Fisher crosses above -1.5 with volume confirmation AND 12h EMA trend up
- Short when Fisher crosses below +1.5 with volume confirmation AND 12h EMA trend down
- Exit when Fisher crosses back through zero
- Volume confirmation: current volume > 1.5x 20-period average
- Works in both bull and bear markets by catching reversals at extremes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_fisher_transform_volume_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Ehlers Fisher Transform (9-period) ===
    # Normalize price to 0-1 range over period
    hlc = (high + low + close) / 3
    n_fisher = 9
    
    # Initialize arrays
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    value = np.full(n, np.nan)
    fisher = np.full(n, np.nan)
    
    for i in range(n_fisher - 1, n):
        # Find highest high and lowest low over the period
        highest[i] = np.max(hlc[i-n_fisher+1:i+1])
        lowest[i] = np.min(hlc[i-n_fisher+1:i+1])
        
        # Avoid division by zero
        if highest[i] != lowest[i]:
            # Normalize to 0-1 range
            value[i] = (hlc[i] - lowest[i]) / (highest[i] - lowest[i])
            # Clamp to avoid extremes
            value[i] = min(0.999, max(0.001, value[i]))
        else:
            value[i] = 0.5
        
        # Fisher transform
        if i >= n_fisher:  # Need previous value
            fisher[i] = 0.5 * np.log((value[i] + 0.999) / (value[i] - 0.999)) + 0.5 * fisher[i-1]
        else:
            fisher[i] = 0.0
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 12h trend filter (EMA 21) ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(n_fisher, n):
        if (np.isnan(fisher[i]) or np.isnan(fisher[i-1]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Fisher crosses back below zero
            if fisher[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Fisher crosses back above zero
            if fisher[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Fisher Transform crossover with volume confirmation AND 12h trend filter
            if fisher[i] > -1.5 and fisher[i-1] <= -1.5 and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                # Fisher crosses above -1.5 with rising 12h EMA -> long
                position = 1
                signals[i] = 0.25
            elif fisher[i] < 1.5 and fisher[i-1] >= 1.5 and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                # Fisher crosses below +1.5 with falling 12h EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals