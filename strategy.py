#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Pivot_Squeeze_Breakout
Hypothesis: Use weekly pivot point support/resistance levels with Bollinger Band squeeze breakout on daily timeframe. 
Weekly pivots provide strong institutional levels. Bollinger Band squeeze indicates low volatility ready for breakout.
Combines mean-reversion (BB) with breakout momentum in a single signal. Works in bull markets (breakouts up) and bear 
markets (breakouts down) by trading direction of breakout from squeeze. Target 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivots to daily timeframe (use previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Bollinger Bands on daily close (20-period, 2 std dev)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band width (normalized) for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    # Squeeze: BB width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for BB calculations
    start_idx = max(40, 20)  # BB(20) + MA(20) for width
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(squeeze[i])):
            signals[i] = 0.0
            continue
        
        pivot_level = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        squeeze_val = squeeze[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Look for Bollinger Band breakout with squeeze release
            # Long: price breaks above upper BB with volume and squeeze release
            if close[i] > bb_up and vol_spike_val and squeeze_val:
                # Additional filter: breakout should be above weekly pivot for validity
                if close[i] > pivot_level:
                    signals[i] = size
                    position = 1
                else:
                    signals[i] = 0.0
            # Short: price breaks below lower BB with volume and squeeze release
            elif close[i] < bb_low and vol_spike_val and squeeze_val:
                # Additional filter: breakout should be below weekly pivot for validity
                if close[i] < pivot_level:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or breaks below support
            if close[i] < bb_middle[i] or close[i] < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB or breaks above resistance
            if close[i] > bb_middle[i] or close[i] > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Pivot_Pivot_Squeeze_Breakout"
timeframe = "1d"
leverage = 1.0