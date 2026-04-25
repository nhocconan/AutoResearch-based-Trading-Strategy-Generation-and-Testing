#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
Hypothesis: Donchian breakouts capture momentum, while weekly pivot direction (from 1d HTF) filters for higher probability trades. Volume confirmation ensures participation. Works in bull (long on upper break with up weekly pivot) and bear (short on lower break with down weekly pivot). Targets 12-37 trades/year on 6h timeframe.
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
    
    # Get 1d data for weekly pivot and Donchian calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points on 1d data (using prior week's high/low/close)
    # We'll approximate weekly by using 5-day rolling on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close (5-day lookback on 1d)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Determine weekly trend: price above pivot = up, below = down
    weekly_trend_up = weekly_close > weekly_pivot
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend_up.astype(float))
    
    # Calculate Donchian channels on 6h (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 20)  # Donchian and volume MA both need 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        vol_ma = vol_ma_20[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        weekly_trend = weekly_trend_up_aligned[i]  # 1.0 for up, 0.0 for down
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > Donchian high, weekly trend up, volume confirmation
            long_entry = (curr_close > dc_high) and (weekly_trend > 0.5) and volume_confirm
            # Short: price < Donchian low, weekly trend down, volume confirmation
            short_entry = (curr_close < dc_low) and (weekly_trend < 0.5) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below weekly pivot OR Donchian low
            if curr_close < weekly_pivot_val or curr_close < dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly pivot OR Donchian high
            if curr_close > weekly_pivot_val or curr_close > dc_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0