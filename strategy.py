#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour trading using 1-week Donchian channel breakout with daily volume confirmation.
# Uses weekly price channels to capture major trend breaks and daily volume to filter false breakouts.
# Weekly timeframe reduces noise, daily volume ensures conviction, and 12h timeframe balances
# responsiveness with low trade frequency. Designed to work in both bull and bear markets by
# following established trends while avoiding overextended moves. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-week Donchian channel
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume and its 20-period average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 12-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x daily volume MA (adjusted for 12h)
        # 2 12h periods per day, so daily MA/2 = approximate 12h period MA
        volume_12h_approx_ma = volume_ma_20_1d_aligned[i] / 2
        volume_condition = volume[i] > (volume_12h_approx_ma * 1.5)
        
        # Entry conditions: Donchian breakout with volume confirmation
        # Long when price breaks above weekly Donchian high with volume
        # Short when price breaks below weekly Donchian low with volume
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Donchian_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0