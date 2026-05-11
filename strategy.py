#!/usr/bin/env python3
"""
6h_WeeklyPivot_MeanReversion_v1
Hypothesis: Price tends to revert to the weekly pivot point (mean of prior week's high, low, close) during range-bound markets, with reversals confirmed by RSI extremes. Weekly pivot provides structural support/resistance that works in both bull and bear markets. Mean reversion on 6h timeframe reduces trade frequency vs lower timeframes, minimizing fee drag while capturing swings around weekly equilibrium.
"""

name = "6h_WeeklyPivot_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for pivot calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    # Handle first week
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if pivot data is invalid
        if np.isnan(pivot_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below pivot AND RSI oversold (<30)
            if close[i] < pivot_aligned[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price above pivot AND RSI overbought (>70)
            elif close[i] > pivot_aligned[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above pivot OR RSI overbought (>70)
            if close[i] > pivot_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses below pivot OR RSI oversold (<30)
            if close[i] < pivot_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals