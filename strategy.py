#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme + 4h EMA34 trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 4h EMA34 AND volume > 1.3 * avg_volume(20)
# Short when Williams %R > -20 (overbought) AND price < 4h EMA34 AND volume > 1.3 * avg_volume(20)
# Exit when price crosses 4h EMA34 in opposite direction
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 80-180 total trades over 4 years (20-45/year) for 4h timeframe
# Williams %R provides mean reversion edge in ranging markets, EMA34 filter ensures trend alignment
# Volume confirmation filters weak signals, works in both bull and bear markets via mean reversion

name = "4h_1dWilliamsR_Extreme_4hEMA34_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Get 4h data ONCE before loop for EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    close_series_4h = pd.Series(close_4h)
    ema_34_4h = close_series_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Williams %R to 4h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Align 4h EMA34 to 4h timeframe (itself, but aligned for consistency)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 4h EMA34 AND volume confirmation
            if (williams_r_aligned[i] < -80 and close[i] > ema_34_4h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 4h EMA34 AND volume confirmation
            elif (williams_r_aligned[i] > -20 and close[i] < ema_34_4h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA34 (trend violation)
            if close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 4h EMA34 (trend violation)
            if close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals