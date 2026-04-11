#!/usr/bin/env python3
# 6h_1d_weekly_pivot_volume_breakout_v1
# Strategy: 6-hour breakout at weekly pivot levels (R4/S4) with 1-day volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Weekly pivot levels (R4/S4) act as strong support/resistance. Breakouts above R4 or below S4
# with above-average volume (1-day volume > 20-period average) capture institutional momentum.
# Works in bull markets by catching upward breakouts and in bear markets by capturing breakdowns.
# Uses 1-day volume filter to avoid false breakouts in low-volume environments.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for weekly pivot and volume filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Need at least 5 days (1 week) of data
    high_5d = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot formula: P = (H + L + C) / 3
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    # Weekly support/resistance levels
    # R4 = P + 3*(H - L)  (strongest resistance)
    # S4 = P - 3*(H - L)  (strongest support)
    weekly_r4 = weekly_pivot + 3.0 * (high_5d - low_5d)
    weekly_s4 = weekly_pivot - 3.0 * (high_5d - low_5d)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # 1-day volume filter: current day volume > 20-day average volume
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / (vol_avg_20 + 1e-10)  # Avoid division by zero
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(5, n):  # Start after weekly pivot warmup
        # Skip if any required data is invalid
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        bull_breakout = close[i] > weekly_r4_aligned[i-1]  # Break above weekly R4
        bear_breakout = close[i] < weekly_s4_aligned[i-1]  # Break below weekly S4
        
        # Volume confirmation: current day volume > 2x 20-day average
        vol_confirm = vol_ratio_aligned[i] > 2.0
        
        # Entry logic: breakout + volume confirmation
        if bull_breakout and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakout and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and bear_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals