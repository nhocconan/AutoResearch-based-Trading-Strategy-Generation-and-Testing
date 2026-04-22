#!/usr/bin/env python3
"""
Hypothesis: 6-hour 4x4 High-Low Channel Breakout with 1-week EMA200 trend and volume spike.
Long when price breaks above 4x4 high with 1-week EMA200 rising and volume spike.
Short when price breaks below 4x4 low with 1-week EMA200 falling and volume spike.
Exit when price retests 4x4 mid-channel (average of high and low).
The 4x4 channel (4-period high/low) provides dynamic support/resistance; 
1-week EMA200 filters long-term trend; volume spike confirms institutional participation.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the weekly trend.
"""

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
    
    # Load 1-week data for EMA200 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1-week EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 4x4 high-low channel (4-period high and low)
    high_4 = pd.Series(high).rolling(window=4, min_periods=4).max().values
    low_4 = pd.Series(low).rolling(window=4, min_periods=4).min().values
    mid_4 = (high_4 + low_4) / 2.0  # Mid-channel for exit
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for calculations
        # Skip if data not ready
        if (np.isnan(high_4[i]) or np.isnan(low_4[i]) or np.isnan(mid_4[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 4x4 high with 1-week EMA200 rising and volume spike
            if (close[i] > high_4[i] and 
                ema200_1w_aligned[i] > ema200_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 4x4 low with 1-week EMA200 falling and volume spike
            elif (close[i] < low_4[i] and 
                  ema200_1w_aligned[i] < ema200_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price retests 4x4 mid-channel
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below mid-channel
                if close[i] < mid_4[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above mid-channel
                if close[i] > mid_4[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_4x4_HighLow_Channel_Breakout_1wEMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0