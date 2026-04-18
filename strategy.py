#!/usr/bin/env python3
"""
6h_SuperTrend_1wTrend_Filter
Hypothesis: Uses SuperTrend on 6h for entry signals, filtered by weekly trend (higher timeframe).
Designed to avoid whipsaws in sideways markets and capture strong trends in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.zeros_like(close_1w)
    ema_50_1w[:] = np.nan
    if len(close_1w) >= 50:
        k = 2 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = close_1w[i] * k + ema_50_1w[i-1] * (1 - k)
    
    # Align weekly EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate SuperTrend on 6h (ATR=10, multiplier=3.0)
    atr_period = 10
    atr_mult = 3.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing (same as RMA)
    atr = np.zeros_like(close)
    atr[:] = np.nan
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # SuperTrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (atr_mult * atr)
    lowerband = hl2 - (atr_mult * atr)
    
    supertrend = np.zeros_like(close)
    supertrend[:] = np.nan
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # Entry conditions
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, atr_period)  # Ensure enough data for EMA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above SuperTrend (uptrend) and close above weekly EMA50
            if close[i] > supertrend[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below SuperTrend (downtrend) and close below weekly EMA50
            elif close[i] < supertrend[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below SuperTrend (trend reversal)
            if close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above SuperTrend (trend reversal)
            if close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_SuperTrend_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0