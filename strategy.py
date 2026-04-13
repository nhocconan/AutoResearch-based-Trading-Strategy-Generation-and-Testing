#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation
    # Enter long when price breaks above R3 with volume > 1.5x 20-bar avg
    # Enter short when price breaks below S3 with volume > 1.5x 20-bar avg
    # Exit when price crosses the 1d midpoint (pivot point)
    # Uses 1d HTF for Camarilla levels (stable institutional levels) and 1w for volume regime filter
    # Volume confirmation ensures breakouts have participation
    # Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for volume confirmation (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    cam_high_low = high_1d - low_1d
    camarilla_r3 = close_1d + (cam_high_low * 1.1 / 4)
    camarilla_s3 = close_1d - (cam_high_low * 1.1 / 4)
    camarilla_mid = (high_1d + low_1d) / 2  # pivot point
    
    # Align 1d Camarilla levels to 1d timeframe (no alignment needed, but using helper for consistency)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > camarilla_r3_aligned[i]  # break above R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # break below S3
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and volume_confirmed[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_mid_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_mid_aligned[i])
        
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

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0