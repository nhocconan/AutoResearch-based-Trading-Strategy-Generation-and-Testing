#!/usr/bin/env python3
# 4h_HMA_Crossover_With_Volume_Filter
# Hypothesis: Hull Moving Average (HMA) crossover on 4h with volume confirmation and 12h trend filter.
# HMA reduces lag while maintaining smoothness; crossovers signal trend changes.
# Volume confirmation filters low-conviction moves; 12h EMA ensures alignment with higher timeframe trend.
# Designed for 20-40 trades/year to avoid fee drag, works in bull/bear via trend following with filters.

name = "4h_HMA_Crossover_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate HMA (9-period for responsiveness, 21 for trend)
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.convolve(arr, np.arange(1, half + 1), 'valid') / (half * (half + 1) / 2)
        wma1 = np.convolve(arr, np.arange(1, period + 1), 'valid') / (period * (period + 1) / 2)
        raw = 2 * wma2 - wma1
        # Pad to original length
        padded = np.full_like(arr, np.nan)
        padded[period-1:period-1+len(raw)] = raw
        return pd.Series(padded).ewm(span=sqrt, min_periods=sqrt).mean().values
    
    hma_fast = hma(close, 9)
    hma_slow = hma(close, 21)
    
    # Calculate 12h EMA for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough for HMA and EMA
    start_idx = max(30, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # HMA crossover signals
        hma_cross_up = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_down = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # Trend filter: price relative to 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: HMA bullish cross + uptrend + volume
            if hma_cross_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: HMA bearish cross + downtrend + volume
            elif hma_cross_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: HMA bearish cross or trend breaks
            if hma_cross_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: HMA bullish cross or trend breaks
            if hma_cross_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals