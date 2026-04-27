#!/usr/bin/env python3
"""
12h_EquityCurveMomentum_WeeklyTrend_Filter
Hypothesis: Uses equity curve momentum (price relative to weekly EMA) with 12h momentum confirmation to ride major trends while avoiding counter-trend trades. Designed for low trade frequency (15-25/year) to minimize fee drag in both bull and bear markets. Weekly trend filter ensures alignment with major market regime.
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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 12h momentum (rate of change over 3 periods)
    roc_period = 3
    roc = np.full_like(close, np.nan)
    for i in range(roc_period, n):
        roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA and ROC
    start_idx = max(50, roc_period)
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(roc[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema50_1w_aligned[i]
        momentum = roc[i]
        
        if position == 0:
            # Long: price above weekly EMA AND positive momentum
            if close[i] > weekly_trend and momentum > 0:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA AND negative momentum
            elif close[i] < weekly_trend and momentum < 0:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA OR momentum turns negative
            if close[i] < weekly_trend or momentum <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA OR momentum turns positive
            if close[i] > weekly_trend or momentum >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EquityCurveMomentum_WeeklyTrend_Filter"
timeframe = "12h"
leverage = 1.0