#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Donchian channels from 4h for breakout structure, 1d EMA34 for trend filter, and volume spike for confirmation.
# Designed for 20-50 trades/year to minimize fee drag. Works in bull markets via breakout continuations and in bear markets via breakdown continuations.
# The 1d EMA34 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.

name = "4h_Donchian20_1dEMA34_VolumeSpike_TrendFilter"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate 4h Donchian channels (20-period) from prior completed 4h bar
    # We need to calculate Donchian on 4h data, then align to 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper and lower bands on 4h data
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 4h bars (no look-ahead)
    upper_4h_shifted = np.roll(upper_4h, 1)
    upper_4h_shifted[0] = np.nan
    lower_4h_shifted = np.roll(lower_4h, 1)
    lower_4h_shifted[0] = np.nan
    
    # Align Donchian bands to 4h timeframe (they're already on 4h, but we need to shift for completed bar)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h_shifted)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(upper_4h_aligned[i]) or
            np.isnan(lower_4h_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND above 1d EMA34 AND volume spike
            if close[i] > upper_4h_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND below 1d EMA34 AND volume spike
            elif close[i] < lower_4h_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Donchian OR below 1d EMA34
            if close[i] < lower_4h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian OR above 1d EMA34
            if close[i] > upper_4h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals