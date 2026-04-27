#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h EMA Crossover with 1d Volume Confirmation and 1w Trend Filter
# Fast EMA (21) crossing above/below Slow EMA (55) on 12h timeframe
# Entry confirmed by 1d volume > 1.5x average and 1w EMA (100) trend direction
# Exit on opposite EMA crossover
# Target: 12-37 trades/year with strong trend capture and low whipsaw

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Volume MA (20-period)
    vol_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i - 19:i + 1])
    
    # Calculate 1w EMA100 for trend filter
    ema_period_1w = 100
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period_1w:
        ema_1w[ema_period_1w - 1] = np.mean(close_1w[:ema_period_1w])
        for i in range(ema_period_1w, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period_1w + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period_1w + 1))))
    
    # Calculate 12h EMAs (21 and 55)
    ema_fast = np.full(n, np.nan)
    ema_slow = np.full(n, np.nan)
    ema_fast_period = 21
    ema_slow_period = 55
    
    if n >= ema_fast_period:
        ema_fast[ema_fast_period - 1] = np.mean(close[:ema_fast_period])
        for i in range(ema_fast_period, n):
            ema_fast[i] = (close[i] * (2 / (ema_fast_period + 1)) + 
                           ema_fast[i - 1] * (1 - (2 / (ema_fast_period + 1))))
    
    if n >= ema_slow_period:
        ema_slow[ema_slow_period - 1] = np.mean(close[:ema_slow_period])
        for i in range(ema_slow_period, n):
            ema_slow[i] = (close[i] * (2 / (ema_slow_period + 1)) + 
                           ema_slow[i - 1] * (1 - (2 / (ema_slow_period + 1))))
    
    # Align 1d volume MA to 12h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Align 1w EMA100 to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMAs and aligned data
    start_idx = max(ema_slow_period, ema_fast_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20_1d_aligned[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: price above/below 1w EMA100
        uptrend = price > ema_1w_aligned[i]
        downtrend = price < ema_1w_aligned[i]
        
        if position == 0:
            # Long: Fast EMA crosses above Slow EMA with volume filter and uptrend
            if (ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and 
                vol_filter and uptrend):
                signals[i] = size
                position = 1
            # Short: Fast EMA crosses below Slow EMA with volume filter and downtrend
            elif (ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and 
                  vol_filter and downtrend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Fast EMA crosses below Slow EMA
            if ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Fast EMA crosses above Slow EMA
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA21_55_Crossover_1dVolume_1wEMA100_Trend"
timeframe = "12h"
leverage = 1.0