#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Donchian Breakout with 1d Volume Spike Filter
# Uses weekly Donchian channels (20-period) for major trend structure and breakout signals
# Filters with 1d volume spikes (>2.0x 20-period average) to avoid false breakouts
# Weekly timeframe provides strong trend context suitable for both bull and bear markets
# Volume confirmation reduces whipsaws and focuses on institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_WeeklyDonchian_Breakout_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Weekly Donchian Channels (20-period)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    donchian_upper = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # weekly Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma_1d = vol_ma_20_1d_aligned[i]
        curr_donchian_upper = donchian_upper_aligned[i]
        curr_donchian_lower = donchian_lower_aligned[i]
        
        # Volume confirmation: current 6h volume > 2.0x 1d average volume (strict threshold)
        vol_confirm = curr_volume > 2.0 * curr_vol_ma_1d
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian lower OR volume confirmation fails
            if curr_close < curr_donchian_lower or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian upper OR volume confirmation fails
            if curr_close > curr_donchian_upper or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above weekly Donchian upper + volume confirmation
            if (curr_high > curr_donchian_upper and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian lower + volume confirmation
            elif (curr_low < curr_donchian_lower and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals