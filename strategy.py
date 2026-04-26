#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_VolumeConfirm
Hypothesis: On 6h timeframe, price breaks above/below weekly Donchian(20) channels with volume confirmation (>1.5x average volume) and alignment with weekly pivot levels. 
In bull markets: price breaks above weekly Donchian high with close above weekly pivot → long. 
In bear markets: price breaks below weekly Donchian low with close below weekly pivot → short. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
Requires BTC/ETH edge via weekly structure and volume filters; avoids SOL-only bias by requiring pivot alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian and pivot calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for HTF structure and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 weeks for Donchian
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    weekly_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot points (standard calculation)
    weekly_high_prev = np.roll(weekly_high, 1)
    weekly_low_prev = np.roll(weekly_low, 1)
    weekly_close_prev = np.roll(weekly_close, 1)
    # Set first value to NaN to avoid look-ahead
    weekly_high_prev[0] = np.nan
    weekly_low_prev[0] = np.nan
    weekly_close_prev[0] = np.nan
    
    weekly_pivot = (weekly_high_prev + weekly_low_prev + weekly_close_prev) / 3.0
    
    # Align weekly data to 6h timeframe (completed weekly bars only)
    weekly_donchian_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_donchian_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate average volume for confirmation (20-period SMA on 6h)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian, 1 for pivot)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Get aligned weekly values
        donchian_high = weekly_donchian_high_aligned[i]
        donchian_low = weekly_donchian_low_aligned[i]
        pivot_val = weekly_pivot_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(donchian_high) or np.isnan(donchian_low) or 
            np.isnan(pivot_val) or np.isnan(avg_vol)):
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
        
        # Long logic: price breaks above weekly Donchian high with close above weekly pivot and volume confirmation
        long_condition = (close_val > donchian_high) and (close_val > pivot_val) and volume_confirmed
        # Short logic: price breaks below weekly Donchian low with close below weekly pivot and volume confirmation
        short_condition = (close_val < donchian_low) and (close_val < pivot_val) and volume_confirmed
        
        # Exit logic: price returns to weekly pivot level or opposite breakout
        exit_long = close_val < pivot_val
        exit_short = close_val > pivot_val
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
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