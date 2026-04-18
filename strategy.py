#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Squeeze_Reversal
Hypothesis: In low-volatility regimes (squeeze), price tends to revert to the mean. We use Bollinger Band width to detect squeeze (BBW < 20th percentile). When in a squeeze and price touches Camarilla support/resistance levels (S1/R1 from daily pivot), we take mean-reversion trades: long at S1, short at R1. Volume confirmation (>1.5x 20-period average) ensures legitimacy. This strategy works in both bull and bear markets because it exploits mean reversion in ranging conditions, which occur during consolidation phases of any trend. Target: 20-40 trades/year via strict squeeze condition and level touches.
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
    
    # Get daily data for pivot and squeeze
    df_1d = get_htf_data(prices, '1d')
    
    # Daily pivot points (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)  # Camarilla R1
    s1 = pivot - (range_1d * 1.1 / 12)  # Camarilla S1
    
    # Align daily levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Bollinger Band width on 4h for squeeze detection (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    
    sma = np.full_like(close, np.nan)
    bb_width = np.full_like(close, np.nan)
    
    if len(close) >= bb_period:
        for i in range(bb_period, len(close)):
            sma[i] = np.mean(close[i - bb_period:i])
            bb_std_dev = np.std(close[i - bb_period:i])
            upper = sma[i] + bb_std * bb_std_dev
            lower = sma[i] - bb_std * bb_std_dev
            bb_width[i] = (upper - lower) / sma[i] * 100  # percent width
    
    # Squeeze condition: BB width < 20th percentile of its history
    bb_width_percentile = np.full_like(bb_width, np.nan)
    lookback = 50  # for percentile calculation
    
    if len(bb_width) >= lookback + bb_period:
        for i in range(lookback + bb_period, len(bb_width)):
            historical = bb_width[i - lookback:i]
            valid = historical[~np.isnan(historical)]
            if len(valid) > 0:
                percentile_20 = np.percentile(valid, 20)
                bb_width_percentile[i] = 1.0 if bb_width[i] < percentile_20 else 0.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    volume_ok = np.full_like(volume, np.nan)
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            volume_ok[i] = 1.0 if volume[i] > 1.5 * vol_ma[i] else 0.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period + lookback, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(volume_ok[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze and volume conditions
        in_squeeze = bb_width_percentile[i] == 1.0
        vol_confirm = volume_ok[i] == 1.0
        
        if position == 0:
            # Long at S1 in squeeze with volume
            if in_squeeze and vol_confirm and low[i] <= s1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short at R1 in squeeze with volume
            elif in_squeeze and vol_confirm and high[i] >= r1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or squeeze breaks
            if high[i] >= pivot_aligned[i] or bb_width_percentile[i] == 0.0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or squeeze breaks
            if low[i] <= pivot_aligned[i] or bb_width_percentile[i] == 0.0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Squeeze_Reversal"
timeframe = "4h"
leverage = 1.0