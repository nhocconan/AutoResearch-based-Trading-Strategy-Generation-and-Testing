#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    # Weekly EMA 20
    ema_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Previous day's EMA for crossover detection
    ema_w_prev = np.roll(ema_w_aligned, 1)
    ema_w_prev[0] = np.nan
    
    # Volume average (20-day)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Crossover signals
    cross_up = (close <= ema_w_prev) & (close > ema_w_aligned)
    cross_down = (close >= ema_w_prev) & (close < ema_w_aligned)
    
    # Volume condition: volume > 2 * 20-day average volume
    vol_condition = volume > 2 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Start after we have at least one previous day and volume MA ready
    start_idx = max(1, 19)  # need index 1 for previous, and 19 for vol_ma_20 (0-indexed)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_w_aligned[i]) or np.isnan(ema_w_prev[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_ok = vol_condition[i]
        cup = cross_up[i]
        cdown = cross_down[i]
        
        if position == 0:
            if cup and vol_ok:
                signals[i] = size
                position = 1
            elif cdown and vol_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if cdown and vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            if cup and vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "Daily_WeeklyEMA20_Crossover_VolumeFilter"
timeframe = "1d"
leverage = 1.0