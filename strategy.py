#!/usr/bin/env python3
"""
1d_Weekly_Trend_Following_v1
Hypothesis: Use weekly price action for trend direction and daily price action for entry timing.
Long when price breaks above weekly high and daily close > weekly EMA(8);
Short when price breaks below weekly low and daily close < weekly EMA(8).
Volume confirmation: daily volume > 1.5x 20-day average volume.
This captures major trends with infrequent entries to minimize fee drag while working in both bull and bear markets.
"""
name = "1d_Weekly_Trend_Following_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 8:
        return np.zeros(n)
    
    # Calculate weekly high and low for breakout levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly EMA(8) for trend filter
    weekly_ema8 = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Align weekly data to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_ema8_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema8)
    
    # Volume filter: daily volume > 1.5 * 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(8, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema8_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high AND daily close > weekly EMA8 AND volume filter
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > weekly_ema8_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low AND daily close < weekly EMA8 AND volume filter
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < weekly_ema8_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to weekly EMA8 (trend reversal)
            if position == 1 and close[i] < weekly_ema8_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > weekly_ema8_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals