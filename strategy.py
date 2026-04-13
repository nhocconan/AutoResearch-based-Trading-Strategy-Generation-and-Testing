#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation
    # Enter long when price breaks above 20-day high with volume > 1.5x 20-day avg
    # Enter short when price breaks below 20-day low with volume > 1.5x 20-day avg
    # Exit on opposite Donchian breakout or when price crosses 20-day midpoint
    # Uses 1w HTF for volume confirmation to reduce noise and false breakouts
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for volume confirmation (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Donchian high = rolling max of high, Donchian low = rolling min of low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0  # midpoint
    
    # Align 1d Donchian levels to 1d timeframe (no alignment needed as already 1d)
    # But we keep the pattern for consistency
    donchian_high_aligned = donchian_high  # already 1d
    donchian_low_aligned = donchian_low    # already 1d
    donchian_mid_aligned = donchian_mid    # already 1d
    
    # Volume confirmation: volume > 1.5x 20-day average volume (using 1d volume)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]   # break above 20-day high
        breakout_down = close[i] < donchian_low_aligned[i]  # break below 20-day low
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and volume_confirmed[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and position != -1
        
        # Exit conditions: opposite breakout or midpoint cross
        exit_long = (position == 1 and (close[i] < donchian_mid_aligned[i] or breakout_down))
        exit_short = (position == -1 and (close[i] > donchian_mid_aligned[i] or breakout_up))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0