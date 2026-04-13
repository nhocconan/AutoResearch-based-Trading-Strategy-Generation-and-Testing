#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation
    # Enter long when price breaks above R4 with volume > 2x 20-bar avg volume
    # Enter short when price breaks below S4 with volume > 2x 20-bar avg volume
    # Exit when price crosses the 12h close (midpoint)
    # Uses 12h HTF for Camarilla levels (more stable than 6h) and 6h for entry timing
    # Camarilla levels from 12h provide institutional support/resistance
    # Volume confirmation ensures breakouts have participation
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Get 12h data for Camarilla pivot calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (based on previous bar's OHLC)
    # Camarilla: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    cam_high_low = high_12h - low_12h
    camarilla_r4 = close_12h + (cam_high_low * 1.1 / 2)
    camarilla_s4 = close_12h - (cam_high_low * 1.1 / 2)
    camarilla_mid = close_12h  # midpoint is the 12h close
    
    # Align 12h Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    
    # Volume confirmation: volume > 2x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > camarilla_r4_aligned[i]  # break above R4
        breakout_down = close[i] < camarilla_s4_aligned[i]  # break below S4
        
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

name = "6h_12h_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0