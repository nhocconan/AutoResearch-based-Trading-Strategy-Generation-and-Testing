#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: Combine daily Camarilla pivot levels with volume confirmation on 12h timeframe.
# Go long when price breaks above R4 with volume confirmation, short when breaks below S4.
# Uses 1d Camarilla levels for institutional reference points, volume filter to avoid false breakouts.
# Works in both bull and bear markets by following institutional support/resistance levels.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using typical Camarilla formula based on previous day's range
    prev_close = df_daily['close'].shift(1).values
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 12h timeframe (will use previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Volume filter: volume > 1.5x 24-period average (12 days of 12h bars)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below R3 or volume fails
            if close[i] < r3_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 or volume fails
            if close[i] > s3_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above R4
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below S4
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals