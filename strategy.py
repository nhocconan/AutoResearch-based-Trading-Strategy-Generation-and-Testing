#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily KAMA trend and weekly volume confirmation.
# Uses KAMA (Kaufman Adaptive Moving Average) for trend detection on 12h chart.
# Weekly volume filter confirms institutional participation. 
# Enters long when price crosses above KAMA with volume confirmation.
# Enters short when price crosses below KAMA with volume confirmation.
# Exits when price returns to KAMA or opposite crossover occurs.
# Designed for 12-37 trades/year with adaptive trend following.

name = "12h_kama_weekly_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close prices
    # Efficiency ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate weekly average volume (20-period)
    volume_1w = df_1w['volume'].values
    vol_avg_20 = np.full_like(volume_1w, np.nan, dtype=float)
    for i in range(19, len(volume_1w)):
        vol_avg_20[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align weekly volume to 12h
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):  # Start after KAMA warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * weekly average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Crossover signals
        cross_above = close[i] > kama[i] and close[i-1] <= kama[i-1]
        cross_below = close[i] < kama[i] and close[i-1] >= kama[i-1]
        
        # Exit conditions: price returns to KAMA or opposite crossover
        exit_long = position == 1 and (close[i] <= kama[i] or cross_below)
        exit_short = position == -1 and (close[i] >= kama[i] or cross_above)
        
        # Entry logic
        if cross_above and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif cross_below and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals