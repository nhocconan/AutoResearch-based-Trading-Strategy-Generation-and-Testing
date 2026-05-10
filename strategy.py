#!/usr/bin/env python3
# 1h_Hull_Moving_Average_Crossover_Trend
# Hypothesis: Hull Moving Average (HMA) reduces lag and improves crossover signals. 
# Use 4h HMA(21) for trend direction and 1h HMA(9)/HMA(21) crossover for entry timing. 
# Adds volume confirmation (>1.5x average) and session filter (08-20 UTC) to reduce false signals. 
# Designed for low frequency (~15-35 trades/year) to minimize fee decay in 1h timeframe.

name = "1h_Hull_Moving_Average_Crossover_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def WMA(values, window):
    """Weighted Moving Average."""
    if len(values) < window:
        return np.full_like(values, np.nan)
    weights = np.arange(1, window + 1)
    return np.convolve(values, weights, 'valid') / weights.sum()

def HMA(values, window):
    """Hull Moving Average."""
    half = window // 2
    sqrt = int(np.sqrt(window))
    wma_half = WMA(values, half)
    wma_full = WMA(values, window)
    raw = 2 * wma_half - wma_full
    return WMA(raw, sqrt)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h HMA(21) for trend
    close_4h = df_4h['close'].values
    hma_4h = HMA(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h HMA(9) and HMA(21) for entry signals
    hma_9 = HMA(close, 9)
    hma_21 = HMA(close, 21)
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_9[i]) or np.isnan(hma_21[i]) or 
            np.isnan(hma_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: HMA(9) crosses above HMA(21) with uptrend on 4h and volume
            if (hma_9[i] > hma_21[i] and hma_9[i-1] <= hma_21[i-1] and
                hma_4h_aligned[i] > close[i] and volume_confirm):
                signals[i] = 0.20
                position = 1
            # Enter short: HMA(9) crosses below HMA(21) with downtrend on 4h and volume
            elif (hma_9[i] < hma_21[i] and hma_9[i-1] >= hma_21[i-1] and
                  hma_4h_aligned[i] < close[i] and volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when HMA(9) crosses below HMA(21) or trend fails
            if (hma_9[i] < hma_21[i] and hma_9[i-1] >= hma_21[i-1]) or hma_4h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when HMA(9) crosses above HMA(21) or trend fails
            if (hma_9[i] > hma_21[i] and hma_9[i-1] <= hma_21[i-1]) or hma_4h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals