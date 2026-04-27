#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_ADX_Filter_v2
Hypothesis: Use 4h KAMA for trend direction, filtered by 1d ADX>25 for trending markets only.
Enter long when KAMA turns up and price>KAMA in uptrend (ADX>25). Enter short when KAMA turns down and price<KAMA in downtrend.
Exit on opposite KAMA crossover. Uses 1d ADX to avoid whipsaws in ranging markets. Target 20-30 trades/year.
Works in bull via trend following, bear via avoiding false signals in ranges.
"""

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
    
    # Calculate 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_smoothed = wilders_smoothing(tr, 14)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 14)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate KAMA on 4h data
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, k=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([0])
        volatility = np.concatenate([np.array([0]), volatility])  # align length
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        return kama
    
    kama_vals = kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.where(kama_vals > np.roll(kama_vals, 1), 1, -1)
    kama_dir[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ADX and KAMA
    start_idx = max(30, 10)  # ADX needs ~30 periods
    
    for i in range(start_idx, n):
        # Skip if ADX not ready
        if np.isnan(adx_aligned[i]) or np.isnan(kama_vals[i]) or np.isnan(kama_dir[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        kama_val = kama_vals[i]
        kama_direction = kama_dir[i]
        
        if position == 0:
            # Require strong trend (ADX > 25) to enter
            if adx_val > 25:
                # Long: KAMA turning up and price > KAMA in uptrend
                if kama_direction == 1 and close[i] > kama_val:
                    signals[i] = size
                    position = 1
                # Short: KAMA turning down and price < KAMA in downtrend
                elif kama_direction == -1 and close[i] < kama_val:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: KAMA turns down
            if kama_direction == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA turns up
            if kama_direction == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_With_1d_ADX_Filter_v2"
timeframe = "4h"
leverage = 1.0