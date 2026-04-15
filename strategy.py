#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot + Volume Confirmation + Volume Spike Filter
# Uses weekly pivot levels (from weekly OHLC) as support/resistance. 
# Long when price crosses above weekly pivot (R1) with volume > 1.5x average and volume spike > 2x median volume (last 20 bars).
# Short when price crosses below weekly pivot (S1) with same volume conditions.
# Works in bull markets (breakouts up from pivot) and bear markets (breakdowns down from pivot).
# Volume confirmation reduces false breakouts; volume spike ensures momentum.
# Target: 50-150 total trades over 4 years = 12-37/year.
# Timeframe: 6h, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivot levels to 6h timeframe (using previous week's values to avoid look-ahead)
    r1_1w_shifted = np.roll(r1_1w, 1)
    s1_1w_shifted = np.roll(s1_1w, 1)
    r1_1w_shifted[0] = np.nan
    s1_1w_shifted[0] = np.nan
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w_shifted)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w_shifted)
    
    # Volume spike: current volume > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    volume_spike = volume > (2 * vol_median)
    
    # Volume confirmation: current volume > 1.5x average of last 20 bars
    vol_mean = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirmed = volume > (1.5 * vol_mean)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            continue
        
        # Long entry: price crosses above R1 with volume confirmation and volume spike
        if (close[i] > r1_1w_aligned[i] and
            volume_confirmed[i] and
            volume_spike[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price crosses below S1 with volume confirmation and volume spike
        elif (close[i] < s1_1w_aligned[i] and
              volume_confirmed[i] and
              volume_spike[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse crossover of pivot levels
        elif position == 1 and close[i] < s1_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r1_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Volume_Spike"
timeframe = "6h"
leverage = 1.0