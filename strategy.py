#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_1w_Volume_Filter
Hypothesis: On 1d timeframe, KAMA captures adaptive trend direction. Combined with 1w volume confirmation (volume > 1.5x 20-period average) to filter breakouts, this strategy works in both bull and bear markets by only taking trades in direction of weekly trend. Uses tight entry criteria to limit trades (7-25/year) and avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if np.sum(volatility[i-er_length+1:i+1]) > 0:
            er[i] = np.abs(close[i] - close[i-er_length]) / np.sum(volatility[i-er_length+1:i+1])
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, er_length=10, fast_sc=2, slow_sc=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate weekly volume average
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if KAMA or volume MA not ready
        if np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current weekly volume > 1.5 * 20-period average
        volume_ok = volume_1w[i] > 1.5 * vol_ma_1w_aligned[i]
        
        # Trend filter: price > KAMA for long, price < KAMA for short
        price = prices['close'].iloc[i]
        trend_long = price > kama_1w_aligned[i]
        trend_short = price < kama_1w_aligned[i]
        
        if position == 0:
            # Long: price above KAMA + volume confirmation + uptrend
            if trend_long and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + volume confirmation + downtrend
            elif trend_short and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA or volume dries up
            if not trend_long or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA or volume dries up
            if not trend_short or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_1w_Volume_Filter"
timeframe = "1d"
leverage = 1.0