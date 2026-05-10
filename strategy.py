#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Breakout of Camarilla R1/S1 levels on 4h chart when weekly trend is strong, using 1-week EMA50 as trend filter.
# In strong weekly uptrend (price > weekly EMA50), long at R1 breakout with target at H3.
# In strong weekly downtrend (price < weekly EMA50), short at S1 breakdown with target at L3.
# Uses volume confirmation to avoid low-liquidity whipsaws. Designed for 4h to achieve 20-40 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    rng = high_1d - low_1d
    r1_1d = close_1d + (1.1 * rng) / 12
    s1_1d = close_1d - (1.1 * rng) / 12
    h3_1d = close_1d + (1.1 * rng) / 4
    l3_1d = close_1d - (1.1 * rng) / 4
    
    # Volume confirmation: 24-period average volume on 1d (4 days of 4h data)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_24_1d = mean_arr(df_1d['volume'].values, 24)
    
    # Align 1w trend to 4h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align 1d Camarilla levels to 4h (wait for 1d bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Align 1d volume MA to 4h
    vol_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or \
           np.isnan(l3_1d_aligned[i]) or np.isnan(vol_ma_24_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        # Get weekly close price aligned to 4h
        close_1w_series = pd.Series(close_1w)
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_series.values)
        is_uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        is_downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Volume condition: current 4h volume > 1.5x 24-period 1d average
        volume_condition = volume[i] > 1.5 * vol_ma_24_1d_aligned[i]
        
        if position == 0:
            # Breakout at R1 in uptrend: long when price breaks R1 resistance
            if is_uptrend and close[i] > r1_1d_aligned[i] and volume_condition:
                signals[i] = 0.25
                position = 1
            # Breakdown at S1 in downtrend: short when price breaks S1 support
            elif is_downtrend and close[i] < s1_1d_aligned[i] and volume_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches H3 (target) or weekly trend turns down
            if close[i] >= h3_1d_aligned[i] or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches L3 (target) or weekly trend turns up
            if close[i] <= l3_1d_aligned[i] or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals