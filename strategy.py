#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Triangular Moving Average (TMA) crossover with 1d volume confirmation and ADX trend filter.
# Long when 12h fast TMA crosses above slow TMA, 1d volume > 1.5x 20 EMA, and 1d ADX > 25.
# Short when fast TMA crosses below slow TMA, 1d volume > 1.5x 20 EMA, and 1d ADX > 25.
# Uses volume for momentum confirmation and ADX to ensure trending markets, avoiding chop.
# Designed for moderate trade frequency (target: 20-40/year) to balance signal quality and cost.
# Works in both bull and bear markets by following 12h momentum with volatility filter.
name = "12h_TMA_Crossover_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 1.5
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    up_move = np.zeros_like(high_1d)
    down_move = np.zeros_like(high_1d)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, len(high_1d)):
        up_move[i] = max(high_1d[i] - high_1d[i-1], 0)
        down_move[i] = max(low_1d[i-1] - low_1d[i], 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    atr = smooth_wilder(tr, 14)
    plus_dm = smooth_wilder(up_move, 14)
    minus_dm = smooth_wilder(down_move, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr > 0, 100 * plus_dm / atr, 0)
    minus_di = np.where(atr > 0, 100 * minus_dm / atr, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_wilder(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 12h data for TMA crossover
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # TMA = SMA of SMA
    def sma(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).mean().values
    
    sma_fast = sma(close_12h, 9)
    sma_slow = sma(close_12h, 21)
    tma_fast = sma(sma_fast, 9)
    tma_slow = sma(sma_slow, 21)
    
    tma_fast_aligned = align_htf_to_ltf(prices, df_12h, tma_fast)
    tma_slow_aligned = align_htf_to_ltf(prices, df_12h, tma_slow)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tma_fast_aligned[i]) or np.isnan(tma_slow_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: TMA fast crosses above slow, volume confirmation, ADX > 25
            tma_cross_up = tma_fast_aligned[i] > tma_slow_aligned[i] and tma_fast_aligned[i-1] <= tma_slow_aligned[i-1]
            long_condition = tma_cross_up and vol_confirm_aligned[i] and (adx_aligned[i] > 25)
            # Short condition: TMA fast crosses below slow, volume confirmation, ADX > 25
            tma_cross_down = tma_fast_aligned[i] < tma_slow_aligned[i] and tma_fast_aligned[i-1] >= tma_slow_aligned[i-1]
            short_condition = tma_cross_down and vol_confirm_aligned[i] and (adx_aligned[i] > 25)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TMA fast crosses below slow or ADX drops below 20 (trend weakening)
            tma_cross_down = tma_fast_aligned[i] < tma_slow_aligned[i] and tma_fast_aligned[i-1] >= tma_slow_aligned[i-1]
            if tma_cross_down or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TMA fast crosses above slow or ADX drops below 20 (trend weakening)
            tma_cross_up = tma_fast_aligned[i] > tma_slow_aligned[i] and tma_fast_aligned[i-1] <= tma_slow_aligned[i-1]
            if tma_cross_up or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals