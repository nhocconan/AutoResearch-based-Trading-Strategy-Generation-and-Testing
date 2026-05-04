#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel from 12h for price structure breakout
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.4x 20 EMA volume) filters false breakouts
# Discrete sizing 0.28 targets 50-150 total trades over 4 years for 12h timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_Donchian20_1dEMA50_VolumeConfirm_Balanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough data for Donchian20 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Calculate Donchian(20) from prior completed 12h bar
    # Upper channel = max(high, 20), Lower channel = min(low, 20)
    upper_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 12h bar
    upper_shifted = np.roll(upper_channel, 1)
    lower_shifted = np.roll(lower_channel, 1)
    upper_shifted[0] = np.nan
    lower_shifted[0] = np.nan
    
    # Align to 12h timeframe (already aligned, but keeping for consistency)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_shifted)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND price > 1d EMA50 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.4 * vol_ema_20[i]):
                signals[i] = 0.28
                position = 1
            # Short conditions: price breaks below lower Donchian AND price < 1d EMA50 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.4 * vol_ema_20[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian OR price crosses below 1d EMA50
            if close[i] < lower_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price returns to upper Donchian OR price crosses above 1d EMA50
            if close[i] > upper_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals