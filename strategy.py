#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_V3
Hypothesis: Combine weekly pivot levels (from 1w) with Donchian(20) breakout on 6h, using weekly trend direction as filter. Enter long when price breaks above Donchian(20) high AND weekly pivot shows bullish bias (price above weekly pivot point); enter short when price breaks below Donchian(20) low AND weekly pivot shows bearish bias (price below weekly pivot point). Uses volume confirmation to avoid false breakouts. Designed for low-moderate trade frequency (15-35/year) to capture significant breakouts in both bull and bear markets while minimizing whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period) ===
    # Highest high of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly Pivot Points ===
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate pivot points from weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot Point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Resistance 1 = (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    # Support 1 = (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    
    # Align weekly pivot data to 6s timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === Volume Confirmation ===
    # 6-period volume average
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60  # For Donchian and volume MA
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma_6[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.5x 6-period average
        vol_filter = volume[i] > 1.5 * vol_ma_6[i]
        
        # Breakout conditions
        breakout_long = high[i] > highest_high[i-1]  # Break above prior Donchian high
        breakout_short = low[i] < lowest_low[i-1]    # Break below prior Donchian low
        
        # Weekly pivot bias
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + price above weekly pivot + volume
            if breakout_long and price_above_pivot and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + price below weekly pivot + volume
            elif breakout_short and price_below_pivot and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or loss of momentum
        elif position == 1:
            # Exit when price breaks below Donchian low or loses weekly pivot support
            if low[i] < lowest_low[i-1] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above Donchian high or loses weekly pivot resistance
            if high[i] > highest_high[i-1] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_V3"
timeframe = "6h"
leverage = 1.0