#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_VolumeConfirm
Hypothesis: Combine weekly pivot points (from 1w) as structural support/resistance with 6h Donchian(20) breakouts and volume confirmation (>1.5x average volume). 
Weekly pivots provide major institutional levels that work in both bull and bear markets. 
Donchian breakouts capture momentum when price breaks weekly pivot resistance/support with volume. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
Designed to work in both bull and bear markets via weekly pivot structure and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll use the prior week's data to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Standard pivot point calculation: P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian and volume)
    start_idx = donchian_period  # 20
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        
        # Get aligned weekly pivot levels (from completed weekly bar)
        pw = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        r2 = weekly_r2_aligned[i]
        s2 = weekly_s2_aligned[i]
        r3 = weekly_r3_aligned[i]
        s3 = weekly_s3_aligned[i]
        
        # Get Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Get average volume
        avg_vol = avg_volume[i]
        
        # Skip if any data not ready
        if (np.isnan(pw) or np.isnan(r1) or np.isnan(s1) or np.isnan(r2) or np.isnan(s2) or 
            np.isnan(r3) or np.isnan(s3) or np.isnan(upper_channel) or np.isnan(lower_channel) or 
            np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above Donchian upper channel AND above weekly R1 with volume confirmation
        long_condition = (close_val > upper_channel) and (close_val > r1) and volume_confirmed
        # Short logic: price breaks below Donchian lower channel AND below weekly S1 with volume confirmation
        short_condition = (close_val < lower_channel) and (close_val < s1) and volume_confirmed
        
        # Exit logic: price returns to weekly pivot level (mean reversion to pivot)
        exit_long = close_val < pw
        exit_short = close_val > pw
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0