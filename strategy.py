#!/usr/bin/env python3
"""
4h_KAMA_Direction_With_1dTrend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 4h for trend direction,
filtered by 1-day EMA to avoid counter-trend trades. Works in bull/bear via
trend filter, with volatility-based position sizing to manage drawdown.
Targets 20-40 trades/year to minimize fee drag.
"""

name = "4h_KAMA_Direction_With_1dTrend_Filter"
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
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will fix in loop
    
    # Calculate ER properly with loop to avoid future leakage
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        change_val = np.abs(close[i] - close[i-er_period])
        volatility_val = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if volatility_val > 0:
            er[i] = change_val / volatility_val
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1-day trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volatility filter: ATR-based position sizing
    atr_period = 14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(er_period, atr_period), n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr[i]) or atr[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Dynamic position size based on volatility (inverse volatility scaling)
        base_size = 0.25
        vol_scaling = min(1.0, 0.015 / atr[i])  # target ~1.5% ATR
        size = base_size * vol_scaling
        size = min(size, 0.35)  # cap at 35%
        
        if position == 0:
            # Long: price above KAMA AND above 1-day EMA
            if close[i] > kama[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: price below KAMA AND below 1-day EMA
            elif close[i] < kama[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals