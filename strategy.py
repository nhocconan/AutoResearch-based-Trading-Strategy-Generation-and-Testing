#!/usr/bin/env python3
"""
6h_1D_Weekly_Pivot_Positioning
Hypothesis: Combine daily and weekly pivot levels to identify institutional support/resistance zones.
Long when price is above weekly pivot AND breaks above daily R1 with volume confirmation.
Short when price is below weekly pivot AND breaks below daily S1 with volume confirmation.
Weekly pivot provides higher-timeframe bias; daily R1/S1 provides precise entry.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via weekly pivot filter (avoids counter-trend trades) and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Daily OHLC for R1/S1 calculation
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly OHLC for pivot calculation
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous day's OHLC for daily R1/S1
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Previous week's OHLC for weekly pivot
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # Daily R1/S1: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high_1d - prev_low_1d
    r1_daily = prev_close_1d + range_1d * 1.1 / 12
    s1_daily = prev_close_1d - range_1d * 1.1 / 12
    
    # Weekly pivot point: P = (high + low + close) / 3
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Align all data to 6h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_1d, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_1d, s1_daily)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price above weekly pivot AND breaks above daily R1 with volume
            if (close[i] > weekly_pivot_aligned[i] and 
                close[i] > r1_daily_aligned[i] and 
                vol_confirm and in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND breaks below daily S1 with volume
            elif (close[i] < weekly_pivot_aligned[i] and 
                  close[i] < s1_daily_aligned[i] and 
                  vol_confirm and in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly pivot or below daily R1
            if close[i] < weekly_pivot_aligned[i] or close[i] < r1_daily_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly pivot or above daily S1
            if close[i] > weekly_pivot_aligned[i] or close[i] > s1_daily_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_Weekly_Pivot_Positioning"
timeframe = "6h"
leverage = 1.0