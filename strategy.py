#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d Williams %R + volume confirmation
# Donchian breakouts capture momentum; 1d Williams %R identifies overbought/oversold extremes
# Volume confirmation ensures breakout authenticity with conviction
# Works in bull/bear: Williams %R adapts to higher timeframe momentum extremes
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "4h_1d_williamsr_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):  # 14-period lookback
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close_1d[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50  # Neutral when range is zero
    
    # Align Williams %R data to 4h timeframe (wait for daily close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR Williams %R > -20 (overbought)
            if close[i] < donchian_low[i] or williams_r_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR Williams %R < -80 (oversold)
            if close[i] > donchian_high[i] or williams_r_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + Williams %R filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND Williams %R < -50 (not overbought)
                if close[i] > donchian_high[i] and williams_r_aligned[i] < -50:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND Williams %R > -50 (not oversold)
                elif close[i] < donchian_low[i] and williams_r_aligned[i] > -50:
                    position = -1
                    signals[i] = -0.25
    
    return signals