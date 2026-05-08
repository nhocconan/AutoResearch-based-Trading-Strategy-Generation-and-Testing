#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action based on 1-week pivot levels with volume confirmation.
# Long when price touches weekly S1 support AND 1d volume > 1.5x 20-period average AND price > 12h EMA50 (uptrend).
# Short when price touches weekly R1 resistance AND 1d volume > 1.5x 20-period average AND price < 12h EMA50 (downtrend).
# Exit when price crosses the weekly pivot point (mean reversion to equilibrium).
# Uses weekly pivots for structure, 1d volume for conviction, 12h EMA50 for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_WeeklyPivot_R1S1_Volume_EMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Daily data for volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    
    # Align weekly pivot levels to 12h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Daily volume filter: current volume > 1.5x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # 12h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price near S1 support, volume confirmation, uptrend
            near_support = abs(low[i] - s1_w_aligned[i]) <= (0.005 * s1_w_aligned[i])  # Within 0.5%
            long_cond = near_support and volume_filter[i] and (close[i] > ema50[i])
            
            # Short conditions: price near R1 resistance, volume confirmation, downtrend
            near_resistance = abs(high[i] - r1_w_aligned[i]) <= (0.005 * r1_w_aligned[i])  # Within 0.5%
            short_cond = near_resistance and volume_filter[i] and (close[i] < ema50[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above weekly pivot (mean reversion)
            if close[i] > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below weekly pivot (mean reversion)
            if close[i] < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals