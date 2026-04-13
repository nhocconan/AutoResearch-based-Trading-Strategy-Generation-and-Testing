#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (trend direction) and 1d volume spike (momentum filter)
# Uses 4h for signal direction (reduces trade frequency), 1h only for entry timing.
# Volume spike on 1d confirms institutional interest. Session filter (08-20 UTC) avoids low-volume hours.
# Position size fixed at 0.20 to manage drawdown. Target: 15-35 trades/year.
# Works in bull (breakouts continue) and bear (volume spikes during capitulation/reversals).

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period Donchian channels on 4h
    highest_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    avg_vol_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    highest_20_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_20_4h)
    lowest_20_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_20_4h)
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    for i in range(200, n):
        # Skip if data not ready or outside session
        if (np.isnan(highest_20_4h_aligned[i]) or 
            np.isnan(lowest_20_4h_aligned[i]) or
            np.isnan(avg_vol_20_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_high = close[i] > highest_20_4h_aligned[i]
        breakout_low = close[i] < lowest_20_4h_aligned[i]
        
        # Volume spike condition: current 1h volume > 1.5x 20-day average volume (scaled to 1h)
        # Approximate 1h volume as 1/6 of daily volume (6 x 1h in 1d)
        vol_1h = volume[i]
        vol_threshold = avg_vol_20_1d_aligned[i] / 6.0 * 1.5  # 1.5x average hourly volume
        volume_spike = vol_1h > vol_threshold
        
        # Entry conditions
        long_entry = breakout_high and volume_spike
        short_entry = breakout_low and volume_spike
        
        # Exit conditions: opposite breakout
        exit_long = position == 1 and breakout_low
        exit_short = position == -1 and breakout_high
        
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

name = "1h_4h_donchian_breakout_vol_spike"
timeframe = "1h"
leverage = 1.0