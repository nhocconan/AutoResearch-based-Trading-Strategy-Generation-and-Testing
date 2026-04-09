#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v1
# Hypothesis: Weekly pivot levels (from 1w data) act as strong support/resistance.
# Breakouts above weekly R3 with volume confirmation indicate bullish momentum.
# Breakdowns below weekly S3 with volume confirmation indicate bearish momentum.
# Works in both bull and bear markets by capturing momentum after key level breaks.
# Target: 20-40 trades/year (80-160 over 4 years) with disciplined entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, R2 = P + (H - L), R3 = H + 2*(P - L)
    # S1 = 2*P - H, S2 = P - (H - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r3 = weekly_high + 2.0 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2.0 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Volume confirmation: 20-period average on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below weekly R3 or volume drops
            if close[i] <= weekly_r3_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises back above weekly S3 or volume drops
            if close[i] >= weekly_s3_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly R3 with volume confirmation
            if close[i] > weekly_r3_aligned[i] and volume[i] > vol_threshold[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly S3 with volume confirmation
            elif close[i] < weekly_s3_aligned[i] and volume[i] > vol_threshold[i]:
                position = -1
                signals[i] = -0.25
    
    return signals