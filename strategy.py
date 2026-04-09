#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Donchian breakout with volume confirmation and session filter
# Use 4h/1d for signal direction (trend/regime), 1h only for entry timing precision
# Session filter (08-20 UTC) reduces noise trades
# Discrete sizing 0.20 to control risk and minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_4h_1d_donchian_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if outside trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(highest_high_4h_aligned[i]) or np.isnan(lowest_low_4h_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.3x 1d average volume (adjusted for timeframe)
        # 1h volume vs daily average: approximate 1/6 of daily volume per hour
        volume_confirmed = volume[i] > 1.3 * (avg_volume_1d_aligned[i] / 6.0)
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower band
            if close[i] < lowest_low_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper band
            if close[i] > highest_high_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long on breakout above 4h Donchian high with volume
            if close[i] > highest_high_4h_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.20
            # Enter short on breakdown below 4h Donchian low with volume
            elif close[i] < lowest_low_4h_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.20
    
    return signals