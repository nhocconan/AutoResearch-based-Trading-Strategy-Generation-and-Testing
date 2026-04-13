#!/usr/bin/env python3
"""
4h_1d_Range_Breakout_With_Volume_Confirmation_v2
Hypothesis: 4h price breaks above/below daily range (high-low) with volume confirmation.
Long when price breaks above prior day's high + volume > 1.5x 20-day average.
Short when price breaks below prior day's low + volume > 1.5x 20-day average.
Exit when price returns to prior day's close.
Designed for 4h timeframe to capture breakouts in both bull and bear markets.
Target: 20-50 trades/year per symbol for better generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily data for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's values for today's calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Daily range and key levels
    prev_range = prev_high - prev_low
    prev_high_level = prev_high
    prev_low_level = prev_low
    
    # Volume moving average
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    
    # Align 1d data to 4h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_level)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_level)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(prev_close_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Breakout conditions
        long_breakout = close[i] > prev_high_aligned[i]
        short_breakout = close[i] < prev_low_aligned[i]
        
        # Exit condition: price returns to prior day's close
        long_exit = close[i] < prev_close_aligned[i]
        short_exit = close[i] > prev_close_aligned[i]
        
        if position == 0:
            if long_breakout and vol_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Range_Breakout_With_Volume_Confirmation_v2"
timeframe = "4h"
leverage = 1.0