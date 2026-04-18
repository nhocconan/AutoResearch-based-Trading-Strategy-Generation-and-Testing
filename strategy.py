#!/usr/bin/env python3
"""
6h_WeeklyPivot_Momentum_Breakout
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts above weekly R1 or below weekly S1 with momentum (ROC > 0) and volume confirmation capture institutional flow. Works in bull/bear by only taking breakouts in direction of weekly trend (price > weekly EMA50 for longs, < for shorts). Targets 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard calculation)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Rate of change (momentum) - 3 period
    roc = np.zeros_like(close)
    roc[3:] = (close[3:] - close[:-3]) / close[:-3] * 100
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from weekly EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Momentum and volume confirmation
        mom_confirm = roc[i] > 0  # positive momentum
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakdown_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: uptrend + momentum + volume + breakout above weekly R1
            if uptrend and mom_confirm and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + momentum + volume + breakdown below weekly S1
            elif downtrend and mom_confirm and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, or breakdown below weekly S1 with volume
            if not uptrend or (volume[i] > vol_ma[i] and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, or breakout above weekly R1 with volume
            if not downtrend or (volume[i] > vol_ma[i] and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Momentum_Breakout"
timeframe = "6h"
leverage = 1.0