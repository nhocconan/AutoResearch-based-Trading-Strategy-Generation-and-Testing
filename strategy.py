#!/usr/bin/env python3
"""
6h Donchian breakout with weekly pivot filter and volume confirmation.
Hypothesis: Donchian(20) breakouts aligned with weekly pivot direction capture trend continuation in both bull and bear markets, while volume confirmation filters false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14299_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points and S3/R3 levels
    # Pivot = (H + L + C) / 3
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    # Range = H - L
    range_weekly = high_weekly - low_weekly
    # S3 = Pivot - 2 * Range
    s3 = pivot - 2.0 * range_weekly
    # R3 = Pivot + 2 * Range
    r3 = pivot + 2.0 * range_weekly
    
    # Align to 6h timeframe (shifted by 1 week for completed bars only)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    
    # 6h data for Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (max of 20 for Donchian)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Check exits: price returns to opposite Donchian band
        if position == 1:  # long position
            if low[i] <= low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if high[i] >= high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Donchian breakouts with weekly pivot filter
            # Long: break above upper band when price > R3 (bullish bias)
            # Short: break below lower band when price < S3 (bearish bias)
            long_breakout = high[i] > high_20[i] and close[i] > r3_aligned[i] and vol_confirm[i]
            short_breakout = low[i] < low_20[i] and close[i] < s3_aligned[i] and vol_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals