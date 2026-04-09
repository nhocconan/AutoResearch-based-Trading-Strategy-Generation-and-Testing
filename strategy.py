#!/usr/bin/env python3
# 6h_weekly_pivot_volume_confirmation_v1
# Hypothesis: 6h strategy using weekly pivot points for trend structure,
# volume confirmation on 6h timeframe to ensure institutional participation,
# and price rejection at pivot support/resistance levels for entries.
# Weekly pivot levels (calculated from prior week) act as significant
# support/resistance that price tends to respect. Volume spikes (>2x 20-period
# MA) confirm breakout/continuation validity. Discrete sizing (0.0, ±0.25)
# minimizes fee churn. Target: 15-25 trades/year.
# Uses weekly HTF data called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_confirmation_v1"
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
    
    # Weekly HTF data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Shift by 1 to use prior week's data (no look-ahead)
    pivot_w = (np.roll(high_w, 1) + np.roll(low_w, 1) + np.roll(close_w, 1)) / 3.0
    range_w = np.roll(high_w, 1) - np.roll(low_w, 1)
    
    r1_w = 2 * pivot_w - np.roll(low_w, 1)
    s1_w = 2 * pivot_w - np.roll(high_w, 1)
    r2_w = pivot_w + range_w
    s2_w = pivot_w - range_w
    r3_w = np.roll(high_w, 1) + 2 * (pivot_w - np.roll(low_w, 1))
    s3_w = np.roll(low_w, 1) - 2 * (np.roll(high_w, 1) - pivot_w)
    
    # Align weekly pivot points to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    
    # 6h volume confirmation
    volume_s = pd.Series(volume)
    volume_ma_6h = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(volume_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma_6h[i]
        
        if position == 1:  # Long position
            # Exit: price falls below S1 OR volume dries up
            if close[i] < s1_1w_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above R1 OR volume dries up
            if close[i] > r1_1w_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above R1 with volume
                if close[i] > r1_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below S1 with volume
                elif close[i] < s1_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals