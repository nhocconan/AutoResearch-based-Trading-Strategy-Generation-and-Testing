#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_Volume
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies adaptive trend direction with reduced whipsaw in sideways markets. Combined with volume confirmation and 1-week trend filter, this strategy captures strong momentum moves while avoiding chop. Timeframe: 1d targets 7-25 trades/year with low fee decay, suitable for both bull and bear regimes via adaptive trend filtering.
"""

name = "1d_KAMA_Trend_Filter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close = prices['close'].values
    # KAMA parameters: ER fast=2, slow=30
    change = np.abs(np.diff(close, k=10))  # 10-period change for efficiency
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder - will compute properly
    
    # Proper KAMA calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = direction / volatility_sum
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[9] = close[9]  # start at index 9
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.5x 20-period EMA
    volume = prices['volume'].values
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10 periods) and EMA34 (34 periods)
    start_idx = max(10, 34)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters: price vs KAMA and 1w EMA34
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Long: price above KAMA, weekly uptrend, volume confirmation
            if price_above_kama and weekly_uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, weekly downtrend, volume confirmation
            elif price_below_kama and weekly_downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or weekly trend fails
            if close[i] <= kama[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or weekly trend fails
            if close[i] >= kama[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals