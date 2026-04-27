#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Chaikin Money Flow (CMF) with 1d trend filter (EMA50).
# Long when CMF crosses above 0.05 (bullish accumulation) and price > 1d EMA50.
# Short when CMF crosses below -0.05 (bearish distribution) and price < 1d EMA50.
# Exit when CMF crosses back through zero (distribution/accumulation ends).
# Uses CMF for institutional flow confirmation, targeting 15-30 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Chaikin Money Flow (20-period)
    cmf_period = 20
    mf_multiplier = np.full(n, np.nan)
    mf_volume = np.full(n, np.nan)
    cmf = np.full(n, np.nan)
    
    for i in range(n):
        if high[i] == low[i]:
            mf_multiplier[i] = 0.0
        else:
            mf_multiplier[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        mf_volume[i] = mf_multiplier[i] * volume[i]
    
    for i in range(cmf_period - 1, n):
        mf_volume_sum = np.sum(mf_volume[i - cmf_period + 1:i + 1])
        volume_sum = np.sum(volume[i - cmf_period + 1:i + 1])
        if volume_sum != 0:
            cmf[i] = mf_volume_sum / volume_sum
    
    # CMF previous value for crossover detection
    cmf_prev = np.full(n, np.nan)
    cmf_prev[1:] = cmf[:-1]
    
    # Align 1d EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need CMF(20) and EMA50
    start_idx = max(cmf_period, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cmf[i]) or np.isnan(cmf_prev[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: CMF crosses above 0.05 and price > 1d EMA50
            if (cmf_prev[i] <= 0.05 and cmf[i] > 0.05 and 
                price > ema_1d_aligned[i]):
                signals[i] = size
                position = 1
            # Short: CMF crosses below -0.05 and price < 1d EMA50
            elif (cmf_prev[i] >= -0.05 and cmf[i] < -0.05 and 
                  price < ema_1d_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: CMF crosses below 0 from above
            if cmf_prev[i] >= 0 and cmf[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: CMF crosses above 0 from below
            if cmf_prev[i] <= 0 and cmf[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_CMF20_1dEMA50_Trend_Filter"
timeframe = "6h"
leverage = 1.0