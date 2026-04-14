#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h EMA20 as trend filter with 4h Donchian(20) breakout
# - Long when 12h trend is up (price above 12h EMA20) and price breaks above 4h Donchian high
# - Short when 12h trend is down (price below 12h EMA20) and price breaks below 4h Donchian low
# - Volume confirmation: volume > 1.2x 20-period average to ensure participation
# - Uses 12h EMA20 to determine trend direction, works in both bull and bear markets
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
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
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
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get 12h index for current 4h bar
        # 12h = 3 * 4h bars
        idx_12h = i // 3
        if idx_12h < 1:
            continue
            
        # Previous 12h EMA20 (to avoid look-ahead)
        ema20_prev = ema20_12h[idx_12h-1]
        
        # Create arrays for alignment (constant values for the 12h period)
        ema20_arr = np.full(len(df_12h), ema20_prev)
        
        # Align to 4h timeframe
        ema20_4h = align_htf_to_ltf(prices, df_12h, ema20_arr)[i]
        
        if position == 0:
            # Long: 12h trend up + price breaks above Donchian high + volume
            if (close[i] > ema20_4h and  # 12h trend up
                high[i] > donch_high[i] and  # Break above Donchian high
                volume[i] > vol_ma[i] * 1.2):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: 12h trend down + price breaks below Donchian low + volume
            elif (close[i] < ema20_4h and  # 12h trend down
                  low[i] < donch_low[i] and  # Break below Donchian low
                  volume[i] > vol_ma[i] * 1.2):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below Donchian low or 12h trend turns down
            if low[i] < donch_low[i] or close[i] < ema20_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above Donchian high or 12h trend turns up
            if high[i] > donch_high[i] or close[i] > ema20_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_EMA20_Donchian_Volume"
timeframe = "4h"
leverage = 1.0