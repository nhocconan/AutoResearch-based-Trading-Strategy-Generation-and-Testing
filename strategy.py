#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d/1w regime filter
    # Bull Power = High - EMA13(close)
    # Bear Power = EMA13(close) - Low
    # Long when: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND 1w close > 1w EMA34
    # Short when: Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA50 AND 1w close < 1w EMA34
    # Exit when: Elder Ray signal weakens (Bull Power <= 0 for long, Bear Power <= 0 for short)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years (12-37/year).
    # Elder Ray measures bull/bear power via price relative to EMA, effective in both trends and ranges.
    # 1d EMA50 and 1w EMA34 provide multi-timeframe trend alignment, reducing whipsaw.
    # Works in bull (long when bulls dominate) and bear (short when bears dominate).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for Elder Ray calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_6h = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power_6h = ema13_6h - low_6h   # Bear Power = EMA13 - Low
    
    # Align 6h Elder Ray to lower timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Warmup period: max of all indicator lookbacks
    warmup = max(13, 50, 34) + 5
    
    for i in range(warmup, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray signals with trend filter
        long_signal = (bull_power_aligned[i] > 0) and (bear_power_aligned[i] < 0) and \
                      (close[i] > ema_50_1d_aligned[i]) and \
                      (close[i] > ema_34_1w_aligned[i])
        
        short_signal = (bear_power_aligned[i] > 0) and (bull_power_aligned[i] < 0) and \
                       (close[i] < ema_50_1d_aligned[i]) and \
                       (close[i] < ema_34_1w_aligned[i])
        
        # Exit when Elder Ray signal weakens
        exit_long = (position == 1 and bull_power_aligned[i] <= 0)
        exit_short = (position == -1 and bear_power_aligned[i] <= 0)
        
        # Execute signals
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0