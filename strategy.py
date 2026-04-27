#!/usr/bin/env python3
"""
1d_WeeklyBreakout_Pullback
Hypothesis: Buy pullbacks to weekly EMA21 in uptrend (price above weekly EMA50),
sell rallies to weekly EMA21 in downtrend (price below weekly EMA50).
Uses daily timeframe for entries with weekly trend filter.
Targets 15-25 trades/year to minimize fee decay while capturing trend continuation.
Works in bull via buying dips, works in bear via selling rallies.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA21 and EMA50
    close_1w = df_1w['close'].values
    
    # EMA21
    ema21_period = 21
    ema21_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema21_period:
        ema21_1w[ema21_period - 1] = np.mean(close_1w[:ema21_period])
        multiplier = 2 / (ema21_period + 1)
        for i in range(ema21_period, len(close_1w)):
            ema21_1w[i] = (close_1w[i] * multiplier) + (ema21_1w[i-1] * (1 - multiplier))
    
    # EMA50
    ema50_period = 50
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema50_period:
        ema50_1w[ema50_period - 1] = np.mean(close_1w[:ema50_period])
        multiplier = 2 / (ema50_period + 1)
        for i in range(ema50_period, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * multiplier) + (ema50_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMAs to daily timeframe
    ema21_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        if np.isnan(ema21_aligned[i]) or np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        
        # Trend filter: price relative to weekly EMA50
        uptrend = price > ema50_aligned[i]
        downtrend = price < ema50_aligned[i]
        
        if position == 0:
            # Long: pullback to weekly EMA21 in uptrend
            if uptrend and price <= ema21_aligned[i] * 1.005:  # within 0.5% above EMA21
                signals[i] = 0.25
                position = 1
            # Short: rally to weekly EMA21 in downtrend
            elif downtrend and price >= ema21_aligned[i] * 0.995:  # within 0.5% below EMA21
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend reversal or extended move
            if price < ema50_aligned[i] or price > ema21_aligned[i] * 1.03:  # 3% above EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or extended move
            if price > ema50_aligned[i] or price < ema21_aligned[i] * 0.97:  # 3% below EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBreakout_Pullback"
timeframe = "1d"
leverage = 1.0