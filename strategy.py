#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: Tighten entry conditions to reduce overtrading. Require volume spike AND price must close outside Camarilla band (not just intrabar touch). Add 1d EMA50 trend filter (slower than EMA34) to reduce whipsaws. Long when close > R1 with 1d EMA50 uptrend and volume spike. Short when close < S1 with 1d EMA50 downtrend and volume spike. Exit on opposite band touch. Discrete sizing 0.25. Target: 20-40 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar)
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    R4_4h = np.full(len(close_4h), np.nan)
    S4_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_4h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_4h[i] = close_prev - (range_prev * 1.1 / 12)
            R4_4h[i] = close_prev + (range_prev * 1.1 / 2)
            S4_4h[i] = close_prev - (range_prev * 1.1 / 2)
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    R4_4h_aligned = align_htf_to_ltf(prices, df_4h, R4_4h)
    S4_4h_aligned = align_htf_to_ltf(prices, df_4h, S4_4h)
    
    # Get 1d data for trend filter (EMA50 - slower for fewer whipsaws)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.5x 20-period average (stricter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price CLOSES above R1 with uptrend and volume spike
            long_signal = (close[i] > R1_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i]
            # Short: price CLOSES below S1 with downtrend and volume spike
            short_signal = (close[i] < S1_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price touches S1 (mean reversion)
            exit_signal = (close[i] < S1_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches R1 (mean reversion)
            exit_signal = (close[i] > R1_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0