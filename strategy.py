#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA crossover with 1w trend filter and volume confirmation
# Uses 6h EMA(12) and EMA(26) for crossover signals, filtered by 1w EMA(50) trend direction.
# Volume confirmation (1.5x average) ensures breakout strength. Works in bull by taking
# longs in uptrend, works in bear by taking shorts in downtrend.
# Target: 60-120 total trades over 4 years (15-30/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for EMA crossover
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Load 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA12 and EMA26 on 6h
    ema12_6h = pd.Series(close_6h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26_6h = pd.Series(close_6h).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    ema12_6h_aligned = align_htf_to_ltf(prices, df_6h, ema12_6h)
    ema26_6h_aligned = align_htf_to_ltf(prices, df_6h, ema26_6h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema12_6h_aligned[i]) or np.isnan(ema26_6h_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: EMA12 crosses above EMA26 + volume spike + price above 1w EMA50 (uptrend)
        if (ema12_6h_aligned[i] > ema26_6h_aligned[i] and
            ema12_6h_aligned[i-1] <= ema26_6h_aligned[i-1] and  # crossover confirmation
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: EMA12 crosses below EMA26 + volume spike + price below 1w EMA50 (downtrend)
        elif (ema12_6h_aligned[i] < ema26_6h_aligned[i] and
              ema12_6h_aligned[i-1] >= ema26_6h_aligned[i-1] and  # crossover confirmation
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and ema12_6h_aligned[i] < ema26_6h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema12_6h_aligned[i] > ema26_6h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_EMA_Crossover_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0