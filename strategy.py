#!/usr/bin/env python3
"""
6H_TurtleSoup_Reversal_v1
Hypothesis: Use 1w pivot points for institutional bias and 6h Turtle Soup pattern (false breakout reversal) for entries.
Long when price breaks below prior day's low then reverses above it with bullish weekly bias;
Short when price breaks above prior day's high then reverses below it with bearish weekly bias.
Volume confirmation: current volume > 1.5x 20-period average volume.
Works in both bull/bear markets by fading institutional stop hunts at key levels.
"""
name = "6H_TurtleSoup_Reversal_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot bias (institutional sentiment)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w
    bias_bullish = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    bias_bearish = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Get 1d data for prior day's high/low (Turtle Soup setup)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Prior day's high and low
    prior_high = df_1d['high'].shift(1).values  # Shift to get prior day only
    prior_low = df_1d['low'].shift(1).values
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    
    start_idx = max(20, 30)  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(bias_bullish[i]) or np.isnan(bias_bearish[i]) or 
            np.isnan(prior_high_aligned[i]) or np.isnan(prior_low_aligned[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades (~1.5 days on 6h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Turtle Soup Long: false breakdown below prior low then reversal
            # Condition: price breaks below prior low, then closes back above it
            if (low[i] < prior_low_aligned[i] and  # broke below prior low
                close[i] > prior_low_aligned[i] and  # closed back above (reversal)
                bias_bullish[i] > 0.5 and          # weekly bullish bias
                volume_filter[i]):                 # volume confirmation
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Turtle Soup Short: false breakout above prior high then reversal
            elif (high[i] > prior_high_aligned[i] and  # broke above prior high
                  close[i] < prior_high_aligned[i] and  # closed back below (reversal)
                  bias_bearish[i] > 0.5 and             # weekly bearish bias
                  volume_filter[i]):                    # volume confirmation
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite prior day level
            if position == 1 and close[i] < prior_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > prior_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals