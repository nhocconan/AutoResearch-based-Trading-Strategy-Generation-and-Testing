#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with Volume Spike and Daily Trend Filter.
Trades breakouts above/below daily Camarilla pivot levels (H4/L4) confirmed by volume spikes,
only when daily price is above/below 50-period EMA to filter range-bound conditions.
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
Uses price action at key institutional levels with volume confirmation.
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous day (H4/L4)
    # Using previous day's data to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day values to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    h4 = pivot + (range_val * 1.1 / 2)  # H4 level
    l4 = pivot - (range_val * 1.1 / 2)  # L4 level
    
    # Breakout conditions: price breaks H4/L4 of previous day
    breakout_up = high_1d > h4
    breakout_down = low_1d < l4
    
    # Align signals to 4h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down.astype(float))
    
    # Daily EMA (50-period) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: price above EMA = uptrend, below EMA = downtrend
    uptrend = close_1d > ema_50
    downtrend = close_1d < ema_50
    
    # Align trend signals to 4h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    # Volume spike: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla breakout + volume spike + daily trend
        long_entry = (breakout_up_aligned[i] > 0.5 and 
                      vol_spike_aligned[i] > 0.5 and 
                      uptrend_aligned[i] > 0.5)
        short_entry = (breakout_down_aligned[i] > 0.5 and 
                       vol_spike_aligned[i] > 0.5 and 
                       downtrend_aligned[i] > 0.5)
        
        # Exit when price returns to pivot point
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        
        exit_long = position == 1 and close[i] <= pivot_aligned[i]
        exit_short = position == -1 and close[i] >= pivot_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0