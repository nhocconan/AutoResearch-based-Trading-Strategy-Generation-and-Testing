#!/usr/bin/env python3
"""
6h_WeeklyPivot_Momentum_Breakout
Hypothesis: Weekly pivot points (R1/S1) act as strong support/resistance. Breakouts with momentum (ROC > 0) and volume confirmation capture institutional flow. Works in bull (breakout continuation) and bear (breakdown continuation) regimes. Target: 20-40 trades/year.
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
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) == 0:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12.0
    
    # Align weekly data to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Momentum indicator: Rate of Change over 6 periods (~1.5 days)
    roc_period = 6
    roc = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(roc_period, len(close)):
        if close[i - roc_period] != 0:
            roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period] * 100.0
    
    # Volume confirmation: 20-period average
    vol_ma = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, roc_period)  # ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(roc[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Momentum condition
        mom_up = roc[i] > 0
        mom_down = roc[i] < 0
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_1w_aligned[i]
        breakdown_down = close[i] < s1_1w_aligned[i]
        
        if position == 0:
            # Long: momentum up + volume + breakout above weekly R1
            if mom_up and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: momentum down + volume + breakdown below weekly S1
            elif mom_down and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: momentum reversal or breakdown below weekly S1
            if not mom_up or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum reversal or breakout above weekly R1
            if not mom_down or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Momentum_Breakout"
timeframe = "6h"
leverage = 1.0