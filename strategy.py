#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation.
# 1d Donchian(20) breakout provides clear trend-following entry points.
# Volume confirmation (>2x 20-period average) filters false breakouts.
# Exit when price returns to opposite Donchian band or volume drops below average.
# Designed for 12h timeframe to reduce trade frequency and minimize fee drag.
# Works in both bull and bear markets by capturing strong directional moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = donchian_period  # Need Donchian channels and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian channel breakouts
            # Long: price breaks above upper Donchian channel
            if (close[i] > upper_channel_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian channel
            elif (close[i] < lower_channel_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian channel or volume drops
            if (close[i] < lower_channel_aligned[i] or 
                volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to upper Donchian channel or volume drops
            if (close[i] > upper_channel_aligned[i] or 
                volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0