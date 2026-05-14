#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses Donchian channels from prior completed 6h bar for structure, 12h EMA50 for trend filter
# Volume confirmation (>1.8x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# 12h EMA50 provides higher timeframe trend filter, reducing whipsaw while capturing major moves.
# Donchian breakouts work well in both bull and bear markets when combined with volume and trend filters.

name = "6h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    close_12h = df_12h['close'].values
    close_12h_shifted = np.roll(close_12h, 1)
    close_12h_shifted[0] = np.nan
    ema_50_12h = pd.Series(close_12h_shifted).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) from prior completed 6h bar
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use only prior completed bar
    upper_channel = np.roll(high_20, 1)
    lower_channel = np.roll(low_20, 1)
    upper_channel[0] = np.nan
    lower_channel[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + price above 12h EMA50 + volume spike
            if close[i] > upper_channel[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + price below 12h EMA50 + volume spike
            elif close[i] < lower_channel[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel OR price crosses below 12h EMA50
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel OR price crosses above 12h EMA50
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals