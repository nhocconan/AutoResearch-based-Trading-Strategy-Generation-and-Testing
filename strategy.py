#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Pivot Reversion with Volume Confirmation
# Uses weekly pivot points (R1, S1) as support/resistance. Mean-reversion when price touches S1 in weekly uptrend or R1 in weekly downtrend.
# Volume filter: current volume > 1.5x 20-period average to confirm institutional interest.
# Designed for 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following weekly trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Weekly trend: 20-period EMA on weekly close
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly uptrend: price above EMA20
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        # Weekly downtrend: price below EMA20
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        # Long conditions: price touches or crosses below S1 in weekly uptrend with volume
        if (weekly_uptrend and 
            low[i] <= s1_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price touches or crosses above R1 in weekly downtrend with volume
        elif (weekly_downtrend and 
              high[i] >= r1_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivotReversion_Volume"
timeframe = "1d"
leverage = 1.0