#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Filter
Hypothesis: KAMA adapts to market noise, reducing false signals in ranging markets.
Combined with volume confirmation and weekly trend filter, this should work in both bull and bear.
Target: 10-20 trades/year (40-80 total over 4 years).
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # KAMA on daily close (adaptive moving average)
    # Efficiency Ratio: |price change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    er[1:] = change[1:] / (np.abs(np.diff(close, n=10)) + 1e-10)  # Simplified ER calculation
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema20_1w_aligned[i]
        kama_val = kama[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price above KAMA + volume + weekly uptrend
            if close[i] > kama_val and vol_ok and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: price below KAMA + volume + weekly downtrend
            elif close[i] < kama_val and vol_ok and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA or weekly trend turns down
            if close[i] < kama_val or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above KAMA or weekly trend turns up
            if close[i] > kama_val or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0