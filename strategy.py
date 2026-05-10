#!/usr/bin/env python3
"""
1d_Keltner_Breakout_Trend_Volume
Hypothesis: In strong weekly trends (EMA100), daily Keltner Channel breakouts with volume confirmation capture explosive moves in both bull and bear markets. Weekly EMA100 filters for major trend direction, while daily Keltner breakouts (2x ATR) provide entry timing with volume filter to avoid false signals. Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag on daily timeframe.
"""

name = "1d_Keltner_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA100)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA100 for trend
    ema100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    # Calculate EMA20 for Keltner middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA100 (100) and daily ATR/EMA20 (20)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema100_1w_aligned[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend_1w = close[i] > ema100_1w_aligned[i]
        downtrend_1w = close[i] < ema100_1w_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-day average volume
        vol_ma20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ma20[i] * 1.5
        
        if position == 0:
            # Long entry: close breaks above Keltner upper + weekly uptrend + volume
            if close[i] > kc_upper[i] and uptrend_1w and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: close breaks below Keltner lower + weekly downtrend + volume
            elif close[i] < kc_lower[i] and downtrend_1w and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close crosses below EMA20 or weekly trend fails
            if close[i] < ema20[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close crosses above EMA20 or weekly trend fails
            if close[i] > ema20[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals