# Hypothesis: Using 12h timeframe with 1-week high/low breakout and volume confirmation
# This strategy aims to capture major trend changes with infrequent entries to minimize fee drag.
# 1-week high/low acts as strong support/resistance, and breakouts with volume confirm institutional interest.
# Works in both bull and bear markets by following the dominant trend direction.
# Target: 12-37 trades per year to stay within fee-efficient range.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for weekly high/low (key levels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly high and low from previous week
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    # Align weekly levels to 12h timeframe
    weekly_high = align_htf_to_ltf(prices, df_1w, prev_high_1w)
    weekly_low = align_htf_to_ltf(prices, df_1w, prev_low_1w)
    
    # Volume filter: volume > 1.5x 30-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above weekly high with volume spike
        long_signal = (close[i] > weekly_high[i] and 
                       close[i-1] <= weekly_high[i-1] and 
                       volume_spike[i])
        
        # Short signal: break below weekly low with volume spike
        short_signal = (close[i] < weekly_low[i] and 
                        close[i-1] >= weekly_low[i-1] and 
                        volume_spike[i])
        
        if long_signal and position != 1:
            signals[i] = 0.30
            position = 1
        elif short_signal and position != -1:
            signals[i] = -0.30
            position = -1
        # Exit when price returns to the opposite weekly level (mean reversion within the weekly range)
        elif position == 1 and close[i] < weekly_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > weekly_high[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyHighLow_Breakout_Volume1.5x"
timeframe = "12h"
leverage = 1.0