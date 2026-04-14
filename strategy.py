#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w pivot direction + 6h Donchian breakout with volume confirmation.
# Weekly pivot direction (based on weekly close vs open) filters for trend direction to avoid counter-trend trades.
# Donchian(20) breakout on 6h provides entry with price channel structure.
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# Exit when price returns to opposite Donchian band or trend weakens (weekly pivot direction flips).
# Designed to work in both bull and bear markets by using 1w trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot direction: 1 if weekly close > open (bullish), -1 if close < open (bearish)
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    pivot_dir = np.where(weekly_close > weekly_open, 1, -1)
    
    # Load 6h data ONCE for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    dc_period = 20
    upper_channel = pd.Series(high_6h).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_channel = pd.Series(low_6h).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Align indicators to 6h timeframe
    pivot_dir_aligned = align_htf_to_ltf(prices, df_1w, pivot_dir)
    upper_channel_aligned = align_htf_to_ltf(prices, df_6h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_6h, lower_channel)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_dir_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts in direction of weekly pivot
            # Long: price breaks above upper channel AND weekly pivot bullish AND volume confirmed
            if (close[i] > upper_channel_aligned[i] and 
                pivot_dir_aligned[i] == 1 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower channel AND weekly pivot bearish AND volume confirmed
            elif (close[i] < lower_channel_aligned[i] and 
                  pivot_dir_aligned[i] == -1 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian channel or weekly pivot turns bearish
            if (close[i] < lower_channel_aligned[i] or 
                pivot_dir_aligned[i] == -1):  # Trend reversal
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to upper Donchian channel or weekly pivot turns bullish
            if (close[i] > upper_channel_aligned[i] or 
                pivot_dir_aligned[i] == 1):  # Trend reversal
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wPivot_6hDonchian_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0