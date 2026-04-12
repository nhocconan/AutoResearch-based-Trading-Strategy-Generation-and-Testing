#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_pivot_volume
# Uses weekly Camarilla pivot levels on daily chart for entries.
# Long when price touches or exceeds weekly S3 (strong support) with volume confirmation.
# Short when price touches or exceeds weekly R3 (strong resistance) with volume confirmation.
# Exits when price returns to weekly pivot point (mean reversion).
# Designed for very low trade frequency (<10 trades/year) to minimize fee drag.
# Works in both bull and bear markets via mean reversion at extreme weekly levels.

name = "1d_1w_camarilla_pivot_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels (S3, S2, S1, PP, R1, R2, R3)
    s3 = pivot - (range_1w * 1.1 / 2)
    s2 = pivot - (range_1w * 1.1 / 4)
    s1 = pivot - (range_1w * 1.1 / 6)
    r1 = pivot + (range_1w * 1.1 / 6)
    r2 = pivot + (range_1w * 1.1 / 4)
    r3 = pivot + (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to daily timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # Volume confirmation: volume > 1.5 * 20-day average (daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price touches or exceeds weekly S3 (strong support)
        if low[i] <= s3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.20
        # Short signal: price touches or exceeds weekly R3 (strong resistance)
        elif high[i] >= r3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: price returns to weekly pivot point (mean reversion)
        elif position == 1 and close[i] >= pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals