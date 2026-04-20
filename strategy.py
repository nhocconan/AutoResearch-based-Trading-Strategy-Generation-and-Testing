#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_R3S3_Reversal_With_VolumeFilter
# Hypothesis: Fade extreme weekly pivot levels (R3/S3) with volume confirmation on 6h timeframe.
# Weekly R3/S3 act as strong support/resistance; price often reverses from these levels.
# Volume filter ensures institutional participation. Works in ranging markets by exploiting
# mean reversion at extremes, and in trends by catching overextended moves.
# Target: 20-40 trades/year (~80-160 total over 4 years).

name = "6h_1d_WeeklyPivot_R3S3_Reversal_With_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Weekly range
    range_w = high_w - low_w
    # Weekly R3 and S3 levels
    r3_w = pivot_w + range_w * 1.1
    s3_w = pivot_w - range_w * 1.1
    
    # Align weekly levels to 6h timeframe
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume must be above average for signal
        if not volume_filter[i]:
            # Hold current position but don't add new signals on low volume
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or goes below S3 (strong support) and shows rejection
            if close[i] <= s3_w_aligned[i] * 1.005:  # Within 0.5% of S3
                # Additional confirmation: close above open (bullish candle)
                if close[i] > prices['open'].iloc[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price touches or goes above R3 (strong resistance) and shows rejection
            elif close[i] >= r3_w_aligned[i] * 0.995:  # Within 0.5% of R3
                # Additional confirmation: close below open (bearish candle)
                if close[i] < prices['open'].iloc[i]:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long: exit if price returns to weekly pivot (mean reversion target)
            # or if price reaches R1 (take profit)
            if close[i] >= pivot_w[i] * 0.995:  # Near pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to weekly pivot
            # or if price reaches S1 (take profit)
            if close[i] <= pivot_w[i] * 1.005:  # Near pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals