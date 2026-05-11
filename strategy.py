#!/usr/bin/env python3
"""
1D_7D_LongOnly_TrendFollow_v1
Hypothesis: Uses weekly trend direction to filter daily long entries. Enters long when price breaks above 20-day Donchian channel and weekly trend is up (price > weekly EMA50). Exits when price breaks below 10-day Donchian channel or weekly trend turns down. Designed for low trade frequency (5-15 trades/year) by requiring weekly trend alignment and daily breakouts. Works in bull markets by capturing trends and avoids most trades in bear markets by staying flat when weekly trend is down.
"""

name = "1D_7D_LongOnly_TrendFollow_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Weekly EMA50 for trend filter ---
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily Donchian Channels (20-period for entry, 10-period for exit) ---
    # Entry: 20-day high
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Exit: 10-day low
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(high_20[i]) or np.isnan(low_10[i]) or np.isnan(ema_50_1w_aligned[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high AND weekly trend is up (price > weekly EMA50)
            if close[i] > high_20[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
        else:
            # Exit long: price breaks below 10-day low OR weekly trend turns down (price < weekly EMA50)
            if close[i] < low_10[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals