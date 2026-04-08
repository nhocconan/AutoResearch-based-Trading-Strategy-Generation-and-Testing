#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v1
# Hypothesis: Uses weekly Donchian channels (20-period) for trend direction on 1d timeframe, with volume confirmation (volume > 1.5x 20-period average) to filter breakouts. Enters long on upward breakout, short on downward breakout. Exits when price crosses back into the channel or volume drops below threshold. Designed for 10-25 trades/year on 1d to avoid fee drag. Works in bull/bear via breakout logic with volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly volume average for confirmation
    volume_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly Donchian channels
    upper_channel = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure Donchian channels are ready
    
    for i in range(start_idx, n):
        if np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or np.isnan(vol_avg_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x weekly average
        volume_confirmed = volume[i] > 1.5 * vol_avg_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below lower channel or volume confirmation fails
            if close[i] < lower_channel_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel or volume confirmation fails
            if close[i] > upper_channel_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume confirmation
            if close[i] > upper_channel_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume confirmation
            elif close[i] < lower_channel_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals