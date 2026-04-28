#!/usr/bin/env python3
"""
12h_KAMA_1dTrend_VolumeSqueeze
Hypothesis: 12h KAMA (Kaufman Adaptive Moving Average) identifies trend direction,
filtered by 1d trend (EMA34) and volume squeeze (low volatility breakout).
Goes long when 12h KAMA turns up in 1d uptrend with volume expansion,
short when 12h KAMA turns down in 1d downtrend with volume expansion.
Designed for very low trade frequency (10-20 trades/year) to minimize fee drag
while capturing major trend changes. Works in both bull and bear markets by
following 1d trend direction and using volatility contraction/expansion.
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
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h KAMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # Will fix below
    
    # Proper ER calculation
    er = np.full_like(close_12h, np.nan)
    for i in range(10, len(close_12h)):
        if i >= 10:
            ch = np.abs(close_12h[i] - close_12h[i-10])
            vol = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
            if vol > 0:
                er[i] = ch / vol
            else:
                er[i] = 1.0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Volume squeeze: volume < 50% of 50-period mean (contraction)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_squeeze = volume < (0.5 * vol_ma_50)
    # Volume expansion: volume > 150% of 50-period mean (breakout)
    volume_expansion = volume > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction
        uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
        downtrend = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        
        # KAMA direction change
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        # Entry conditions: KAMA turn in direction of 1d trend with volume expansion
        long_entry = kama_up and uptrend and volume_expansion[i]
        short_entry = kama_down and downtrend and volume_expansion[i]
        
        # Exit conditions: opposite KAMA turn or loss of trend
        long_exit = kama_down or (not uptrend)
        short_exit = kama_up or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_1dTrend_VolumeSqueeze"
timeframe = "12h"
leverage = 1.0