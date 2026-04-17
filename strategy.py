#!/usr/bin/env python3
"""
1d_PowerTrend_Reversal
Strategy: Buy near weekly pivot S1/S2 support in uptrend, sell near R1/R2 resistance in downtrend.
Uses weekly pivot levels from 1w data, 1d EMA50 trend filter, and volume confirmation.
Only trades when price is near weekly support/resistance levels with trend alignment.
Designed to work in both bull and bear markets by trading mean reversion at key levels.
Timeframe: 1d
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
    
    # Calculate 1d EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for pivot calculation (done once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points
    typical_price = (high_1w + low_1w + close_1w) / 3
    range_val = high_1w - low_1w
    
    # Pivot point and support/resistance levels
    pp = typical_price
    r1 = (2 * pp) - low_1w
    s1 = (2 * pp) - high_1w
    r2 = pp + range_val
    s2 = pp - range_val
    
    # Align weekly levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        # Price proximity to weekly levels (within 0.75%)
        near_s1 = abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.0075
        near_s2 = abs(close[i] - s2_aligned[i]) / s2_aligned[i] < 0.0075
        near_r1 = abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.0075
        near_r2 = abs(close[i] - r2_aligned[i]) / r2_aligned[i] < 0.0075
        
        if position == 0:
            # Long: near weekly support (S1 or S2) + uptrend + volume
            if (near_s1 or near_s2) and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: near weekly resistance (R1 or R2) + downtrend + volume
            elif (near_r1 or near_r2) and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price moves near weekly resistance or trend changes
            if (near_r1 or near_r2) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves near weekly support or trend changes
            if (near_s1 or near_s2) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_PowerTrend_Reversal"
timeframe = "1d"
leverage = 1.0