#!/usr/bin/env python3
"""
1d_1w_Range_Breakout_With_Volume
Hypothesis: On daily timeframe, price breaking out of the prior week's range with volume confirmation
captures institutional flow. In bull markets, breaks above weekly high sustain; in bear markets,
breaks below weekly low sustain. Uses 1-week range as dynamic support/resistance. Volume filter
avoids false breakouts. Designed for low trade frequency (<15/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 5:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high and low (from completed weekly bars)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly high/low to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(5, n):
        # Skip if weekly data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: break above weekly high with volume
        long_breakout = (close[i] > weekly_high_aligned[i]) and volume_expansion[i]
        # Short: break below weekly low with volume
        short_breakout = (close[i] < weekly_low_aligned[i]) and volume_expansion[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Range_Breakout_With_Volume"
timeframe = "1d"
leverage = 1.0