#!/usr/bin/env python3
"""
Hypothesis: 1-day Exponential Moving Average crossover with weekly trend filter and volume confirmation.
Goes long when price crosses above EMA(21) and weekly EMA(21) is rising, short when price crosses below EMA(21) and weekly EMA(21) is falling.
Uses volume > 1.5x 20-day average to confirm strength.
Designed to work in both bull and bear markets by using the weekly trend as filter.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA(21) for entry signal
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily volume MA(20) for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA(21), weekly EMA aligned, and volume MA
    start_idx = max(21, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema_fast = ema_21[i]
        ema_slow_weekly = ema_21_1w_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: EMA crossover with weekly trend and volume confirmation
        if position == 0:
            # Long: price crosses above daily EMA(21) + weekly EMA rising + volume
            if close[i] > ema_fast and ema_slow_weekly > ema_21_1w_aligned[max(i-1, start_idx)] and vol_filter:
                signals[i] = size
                position = 1
            # Short: price crosses below daily EMA(21) + weekly EMA falling + volume
            elif close[i] < ema_fast and ema_slow_weekly < ema_21_1w_aligned[max(i-1, start_idx)] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below daily EMA(21)
            if close[i] < ema_fast:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above daily EMA(21)
            if close[i] > ema_fast:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA21_Crossover_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0