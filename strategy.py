#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Donchian_Breakout_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20 periods)
    weekly_high = pd.Series(df_weekly['high']).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(df_weekly['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h timeframe
    weekly_high_aligned = align_ltf_to_htf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_ltf_to_htf(prices, df_weekly, weekly_low)
    
    # Weekly pivot points from previous week
    weekly_high_prev = pd.Series(df_weekly['high']).shift(1).values
    weekly_low_prev = pd.Series(df_weekly['low']).shift(1).values
    weekly_close_prev = pd.Series(df_weekly['close']).shift(1).values
    
    # Calculate pivot points (standard formula)
    pivot = (weekly_high_prev + weekly_low_prev + weekly_close_prev) / 3.0
    r1 = 2 * pivot - weekly_low_prev
    s1 = 2 * pivot - weekly_high_prev
    r2 = pivot + (weekly_high_prev - weekly_low_prev)
    s2 = pivot - (weekly_high_prev - weekly_low_prev)
    r3 = weekly_high_prev + 2 * (pivot - weekly_low_prev)
    s3 = weekly_low_prev - 2 * (weekly_high_prev - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_ltf_to_htf(prices, df_weekly, pivot)
    r1_aligned = align_ltf_to_htf(prices, df_weekly, r1)
    s1_aligned = align_ltf_to_htf(prices, df_weekly, s1)
    r2_aligned = align_ltf_to_htf(prices, df_weekly, r2)
    s2_aligned = align_ltf_to_htf(prices, df_weekly, s2)
    r3_aligned = align_ltf_to_htf(prices, df_weekly, r3)
    s3_aligned = align_ltf_to_htf(prices, df_weekly, s3)
    
    # Volume filter: 6h volume > 1.5x 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24)  # Wait for weekly Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or \
           np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly R3 with volume spike
            if close[i] > r3_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S3 with volume spike
            elif close[i] < s3_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly pivot or weekly low
            if close[i] < pivot_aligned[i] or close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly pivot or weekly high
            if close[i] > pivot_aligned[i] or close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points act as strong support/resistance levels.
# Breakouts above R3 or below S3 with volume confirmation indicate strong momentum.
# Uses weekly Donchian channels for trend context and weekly pivot levels for entry/exit.
# Volume spike (>1.5x average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support).
# Target: 15-35 trades/year to minimize fee decay while capturing significant moves.