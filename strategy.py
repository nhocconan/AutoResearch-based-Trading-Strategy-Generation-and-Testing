#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot point rejection + 1d volume confirmation
# Short when price rejects weekly R3 (fails to close above) AND volume > 1.5 * avg_volume(20)
# Long when price rejects weekly S3 (fails to close below) AND volume > 1.5 * avg_volume(20)
# Uses weekly pivot levels from prior completed week for structure
# Volume confirmation filters weak rejections
# Discrete sizing 0.25 to balance return and drawdown
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe
# Weekly pivots provide significant support/resistance that price often respects
# Rejection at these levels with volume indicates institutional interest
# Works in ranging markets (fade extremes) and can catch reversals in trends

name = "6h_WeeklyPivotRejection_VolumeConfirm"
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
    
    # Get weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one completed week
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (based on prior completed week)
    # Standard pivot: P = (H + L + C) / 3
    # R3 = P + 2*(H - L), S3 = P - 2*(H - L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    camarilla_r3_1w = pivot_1w + 2.0 * (high_1w - low_1w)
    camarilla_s3_1w = pivot_1w - 2.0 * (high_1w - low_1w)
    
    # Align weekly pivot points to 6h timeframe (wait for completed weekly bar)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20_1d)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i]) or 
            np.isnan(avg_volume_20_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects weekly S3 (closes above it) with volume confirmation
            # Rejection = price moves above S3 but doesn't sustain below it
            # We enter when we see confirmation of rejection: close > S3 AND volume spike
            if close[i] > camarilla_s3_1w_aligned[i] and volume[i] > (1.5 * avg_volume_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price rejects weekly R3 (closes below it) with volume confirmation
            elif close[i] < camarilla_r3_1w_aligned[i] and volume[i] > (1.5 * avg_volume_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly S3 (rejection failed) OR volume drops
            if close[i] < camarilla_s3_1w_aligned[i] or volume[i] < avg_volume_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly R3 (rejection failed) OR volume drops
            if close[i] > camarilla_r3_1w_aligned[i] or volume[i] < avg_volume_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals