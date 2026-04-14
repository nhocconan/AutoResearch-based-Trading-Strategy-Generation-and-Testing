#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d EMA100 as trend filter with 4h Donchian(20) breakout
# - Long when 1d trend is up (close > 1d EMA100) and price breaks above 4h Donchian high
# - Short when 1d trend is down (close < 1d EMA100) and price breaks below 4h Donchian low
# - Volume confirmation: volume > 1.5x 20-period average to ensure participation
# - Uses 1d EMA100 for stronger trend filtering, reducing whipsaw in sideways markets
# - Donchian(20) provides clear breakout levels with proper risk management
# - Target: 100-180 total trades over 4 years (25-45/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d EMA100
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get 1d index for current 4h bar
        # 1d = 6 * 4h bars
        idx_1d = i // 6
        if idx_1d < 1:
            continue
            
        # Previous 1d EMA100 (to avoid look-ahead)
        ema100_prev = ema100_1d[idx_1d-1]
        
        # Create arrays for alignment (constant values for the 1d period)
        ema100_arr = np.full(len(df_1d), ema100_prev)
        
        # Align to 4h timeframe
        ema100_4h = align_htf_to_ltf(prices, df_1d, ema100_arr)[i]
        
        if position == 0:
            # Long: 1d trend up + price breaks above Donchian high + volume
            if (close[i] > ema100_4h and  # 1d trend up
                high[i] > donch_high[i] and  # Break above Donchian high
                volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: 1d trend down + price breaks below Donchian low + volume
            elif (close[i] < ema100_4h and  # 1d trend down
                  low[i] < donch_low[i] and  # Break below Donchian low
                  volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below Donchian low or 1d trend turns down
            if low[i] < donch_low[i] or close[i] < ema100_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above Donchian high or 1d trend turns up
            if high[i] > donch_high[i] or close[i] > ema100_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_EMA100_Donchian_Volume"
timeframe = "4h"
leverage = 1.0