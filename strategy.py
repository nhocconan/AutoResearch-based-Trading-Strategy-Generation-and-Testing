#!/usr/bin/env python3
"""
1d_KAMA_Reversal_With_Trend_Filter
Hypothesis: KAMA (adaptive moving average) identifies trend direction and potential reversals.
Combined with 1-week EMA trend filter and volume confirmation to avoid false signals.
Designed to work in both bull and bear markets by using adaptive smoothing and strict entry conditions.
Target: 15-25 trades per year (60-100 total over 4 years).
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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 14-period
    # Efficiency Ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of abs changes over 10 periods
    # Pad volatility to match change length
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    
    # Calculate ER with proper handling
    er = np.full_like(close, np.nan, dtype=float)
    valid_idx = ~np.isnan(volatility_padded) & (volatility_padded != 0)
    er[valid_idx] = change / volatility_padded[valid_idx]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week EMA20 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period - need enough data for KAMA calculation
    start_idx = 30  # 10 for ER + 20 for volatility + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume spike and weekly uptrend
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and 
                volume_spike[i] and close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume spike and weekly downtrend
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and 
                  volume_spike[i] and close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below KAMA or weekly trend fails
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or weekly trend fails
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Reversal_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0