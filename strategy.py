#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Range with Volume Confirmation
# Hypothesis: Price tends to respect weekly pivot support/resistance levels.
# Buy near weekly S1/S2 with bullish volume, sell near R1/R2 with bearish volume.
# Weekly pivot provides structure that works in both bull and bear markets.
# Volume confirmation filters false breaks. Target: 15-25 trades/year (60-100 total).

name = "6h_weekly_pivot_range_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3
    # Support and Resistance levels
    s1 = (2 * pp) - weekly_high
    s2 = pp - (weekly_high - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r1 = (2 * pp) - weekly_low
    r2 = pp + (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R1 or volume dries up
            if high[i] >= r1_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches S1 or volume dries up
            if low[i] <= s1_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long near support with bullish volume
            if (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short near resistance with bearish volume
            elif (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals