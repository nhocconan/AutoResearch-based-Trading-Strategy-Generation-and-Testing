#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot levels as trend filter and daily volume spike for entry.
# Weekly pivot levels (calculated from prior week) provide robust trend bias less prone to whipsaw.
# Daily volume spike confirms institutional participation, reducing false breakouts.
# This combination should work in both bull and bear markets by filtering for high-probability
# institutional moves aligned with the weekly trend. Target: 15-30 trades/year.

name = "6h_WeeklyPivot_Trend_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    
    # Daily volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly pivot levels to 6h
    pivot_w_6h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_6h = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_6h = align_htf_to_ltf(prices, df_1w, s2_w)
    r3_w_6h = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_w_6h = align_htf_to_ltf(prices, df_1w, s3_w)
    
    # Align daily volume average to 6h
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_w_6h[i]) or np.isnan(r1_w_6h[i]) or np.isnan(s1_w_6h[i]) or
            np.isnan(r2_w_6h[i]) or np.isnan(s2_w_6h[i]) or np.isnan(r3_w_6h[i]) or
            np.isnan(s3_w_6h[i]) or np.isnan(vol_avg_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly pivot
        above_pivot = close[i] > pivot_w_6h[i]
        below_pivot = close[i] < pivot_w_6h[i]
        
        # Volume confirmation: current volume > 1.5x daily average
        vol_spike = volume[i] > vol_avg_1d_6h[i] * 1.5
        
        if position == 0:
            # Long: price above weekly pivot with volume spike
            if above_pivot and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot with volume spike
            elif below_pivot and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly pivot
            if below_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly pivot
            if above_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals