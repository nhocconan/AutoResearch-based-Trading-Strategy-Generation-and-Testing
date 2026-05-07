# 175336
#!/usr/bin/env python3
name = "6h_12h_Donchian20_Slope_Filter_Trend"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h Donchian channel (20 periods)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling high/low using pandas
    high_roll = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian slope (change over 3 periods)
    high_slope = np.diff(high_roll, prepend=high_roll[0])
    low_slope = np.diff(low_roll, prepend=low_roll[0])
    high_slope_3 = pd.Series(high_slope).rolling(window=3, min_periods=1).mean().values
    low_slope_3 = pd.Series(low_slope).rolling(window=3, min_periods=1).mean().values
    
    # Align to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_roll)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_roll)
    high_slope_aligned = align_htf_to_ltf(prices, df_12h, high_slope_3)
    low_slope_aligned = align_htf_to_ltf(prices, df_12h, low_slope_3)
    
    # 6h volume spike detection (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(high_slope_aligned[i]) or np.isnan(low_slope_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper band with upward slope and volume
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            if close[i] > high_20_aligned[i] and high_slope_aligned[i] > 0 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band with downward slope and volume
            elif close[i] < low_20_aligned[i] and low_slope_aligned[i] < 0 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below upper band or slope turns down
            if close[i] < high_20_aligned[i] or high_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above lower band or slope turns up
            if close[i] > low_20_aligned[i] or low_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian breakout with 12h slope filter and volume confirmation
# - Uses 12h Donchian channel (20-period) for major support/resistance levels
# - Adds slope filter to ensure breakout occurs in direction of channel momentum
# - Volume confirmation (1.5x average) filters false breakouts
# - Works in both bull and bear markets by trading breakouts in direction of 12h momentum
# - Position size 0.25 targets ~50-100 trades over 4 years (12-25/year)
# - Slope filter reduces whipsaw by requiring momentum alignment
# - Designed for 6h timeframe to balance signal frequency and transaction costs