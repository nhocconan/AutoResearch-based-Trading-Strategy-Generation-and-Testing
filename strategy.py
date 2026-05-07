#!/usr/bin/env python3
name = "4h_4H_KAMA_Trend_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros(n)
    er[er_len:] = change[er_len:] / np.maximum(volatility[er_len:], 1e-10)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA20 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Donchian(20) channels from 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Volume spike: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Wait for channels and KAMA
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above upper Donchian with volume spike in uptrend
            if close[i] > upper_aligned[i] and vol_spike[i] and close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian with volume spike in downtrend
            elif close[i] < lower_aligned[i] and vol_spike[i] and close[i] < ema20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below KAMA or trend turns down
            if close[i] < kama[i] or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above KAMA or trend turns up
            if close[i] > kama[i] or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend filter with Donchian breakout and volume confirmation.
# Long when price breaks above 1d Donchian upper channel with volume spike in 1d uptrend.
# Short when price breaks below 1d Donchian lower channel with volume spike in 1d downtrend.
# Uses KAMA(10,2,30) as trend filter to avoid whipsaws. Exit when price crosses KAMA or trend fails.
# Volume spike (>2.0x average) ensures conviction. Designed for 4h timeframe targeting 20-40 trades/year.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# KAMA adapts to market noise, reducing false signals in choppy conditions.