#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d EMA50 as trend filter with 4h Donchian(15) breakout
# - Long when 1d trend is up (close > 1d EMA50) and price breaks above 4h Donchian high
# - Short when 1d trend is down (close < 1d EMA50) and price breaks below 4h Donchian low
# - Volume confirmation: volume > 1.3x 20-period average to ensure participation
# - Uses 1d EMA50 to determine trend direction, works in both bull and bear markets
# - Donchian breakout provides entry timing with clear risk management
# - Target: 75-200 total trades over 4 years (19-50/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h Donchian channels (15-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=15, min_periods=15).max().values
    donch_low = low_series.rolling(window=15, min_periods=15).min().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get 1d index for current 4h bar
        # 1d = 6 * 4h bars
        idx_1d = i // 6
        if idx_1d < 1:
            continue
            
        # Previous 1d EMA50 (to avoid look-ahead)
        ema50_prev = ema50_1d[idx_1d-1]
        
        # Create arrays for alignment (constant values for the 1d period)
        ema50_arr = np.full(len(df_1d), ema50_prev)
        
        # Align to 4h timeframe
        ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_arr)[i]
        
        if position == 0:
            # Long: 1d trend up + price breaks above Donchian high + volume
            if (close[i] > ema50_4h and  # 1d trend up
                high[i] > donch_high[i] and  # Break above Donchian high
                volume[i] > vol_ma[i] * 1.3):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: 1d trend down + price breaks below Donchian low + volume
            elif (close[i] < ema50_4h and  # 1d trend down
                  low[i] < donch_low[i] and  # Break below Donchian low
                  volume[i] > vol_ma[i] * 1.3):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below Donchian low or 1d trend turns down
            if low[i] < donch_low[i] or close[i] < ema50_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above Donchian high or 1d trend turns up
            if high[i] > donch_high[i] or close[i] > ema50_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_EMA50_Donchian_Volume"
timeframe = "4h"
leverage = 1.0