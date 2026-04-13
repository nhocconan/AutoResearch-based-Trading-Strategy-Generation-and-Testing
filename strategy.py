#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakouts with 1d volume regime filter
    # Long when: price breaks above H4 (Camarilla resistance) AND 1d volume > 20-day average
    # Short when: price breaks below L4 (Camarilla support) AND 1d volume > 20-day average
    # Exit when: price returns to PIVOT point (mid-price)
    # Uses discrete sizing (0.30) targeting 50-150 total trades over 4 years (12-37/year).
    # 12h timeframe minimizes fee drag. Camarilla levels from 1d provide structure.
    # Volume regime filter ensures breakouts occur during institutional participation.
    # Works in bull (breakouts with volume) and bear (only volume-confirmed breaks taken).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for price action (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for Camarilla calculations (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    camarilla_high = (high_1d + low_1d + 2 * close_1d) / 4  # PIVOT
    camarilla_range = high_1d - low_1d
    camarilla_h4 = camarilla_high + 1.5 * camarilla_range  # Resistance
    camarilla_l4 = camarilla_high - 1.5 * camarilla_range  # Support
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    
    # Calculate volume regime: 1d volume > 1.5x 20-day average
    avg_volume_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume > (1.5 * avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.30  # 30% position size
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > camarilla_h4_aligned[i-1]  # break above H4
        breakout_down = close[i] < camarilla_l4_aligned[i-1]  # break below L4
        
        # Entry conditions with volume regime filter
        long_entry = breakout_up and volume_regime[i] and position != 1
        short_entry = breakout_down and volume_regime[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_pivot_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_pivot_aligned[i])
        
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

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0